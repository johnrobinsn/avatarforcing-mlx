# MLX port of AvatarForcing (https://github.com/TaekyungKi/AvatarForcing), an adaptation of
# models/avatarforcing/flow_transformer.py at commit 507e3046ad53. Copyright the AvatarForcing authors; MLX port by
# John Robinson. Licensed CC BY-NC 4.0 (see LICENSE and NOTICE).

"""MLX port of AvatarForcing flow_transformer.

Module tree (mirrors PyTorch `flow_transformer.py` 1:1):

    FlowTransformerMLX
        .x_embedder        : SequenceEmbed (Linear 1024 -> 1024)
        .t_embedder        : TimestepEmbedder (sinusoidal -> MLP)
        .c_embedder        : AudioMotionAttention (motion MLP + 2× MHA + Linear)
        .adaLN_modulation  : [SiLU, Linear hidden -> 6*hidden]
        .blocks[0..7]      : DFoTBlock
            .attn          : Attention (self-attn, RoPE, block-bidir mask, 38-slot KV cache)
            .cross_attn    : CrossAttention (windowed cross-attn, 2-slot KV cache)
            .mlp           : Mlp (hidden -> 4*hidden -> hidden, GELU-tanh)
            .norm1/2/3     : LayerNorm (elementwise_affine=False) — parameterless
        .final_layer       : FinalLayer (adaLN + Linear hidden -> dim_w)

Weight-key convention: identical to the PyTorch state_dict (`blocks.0.attn.qkv.weight`,
`t_embedder.mlp.0.weight`, ...).  PyTorch nn.Sequential containers are mirrored
with plain Python lists (MLX's nn.Sequential would add a `.layers.` level to
every key).  The only conversion step is splitting nn.MultiheadAttention's
packed in_proj into query/key/value projections — see convert.py.

KV cache: same structure as PyTorch — `kv_cache_list[i] = (self_cache, cross_cache)`
where each cache is a dict {"k": array, "v": array}.  Updates rebind the dict
entries (MLX arrays are immutable), which is the functional equivalent of the
in-place writes in PyTorch.

RoPE: `forward_with_precomputed` takes `start_pos` (int) instead of a pre-sliced
`freqs_cis_slice`; rotation uses `mx.fast.rope(traditional=True, offset=start_pos)`.

Hyperparameters (from configs/inference.yaml):
    dim_w = 512, dim_h = 1024, dim_a = 512, num_heads = 8, transformer_depth = 8,
    num_frames_for_clip = 50, num_frames_per_block = 10, attention_window = 2,
    mlp_ratio = 4.0
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .helpers import (
    cross_mask_local,
    enc_dec_mask,
    make_block_bidir_causal_lookahead_mask,
    modulate,
)


def _layer_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """LayerNorm(elementwise_affine=False)."""
    return mx.fast.layer_norm(x, None, None, eps)


# =====================================================================
#  Small embedders
# =====================================================================

class SequenceEmbed(nn.Module):
    """nn.Linear with an identity post-norm.  PyTorch: line 361."""
    def __init__(self, dim_w: int, dim_h: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_w, dim_h, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + MLP.  PyTorch: line 627."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = [
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half).astype(mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        for layer in self.mlp:
            x = layer(x)
        return x


# =====================================================================
#  AudioMotionAttention (c_embedder)
# =====================================================================

class AudioMotionAttention(nn.Module):
    """PyTorch: line 97.  Fuses user audio (kv), user motion (q) and agent audio (q2)."""
    def __init__(self, dim_a: int, dim_h: int):
        super().__init__()
        self.motion_embedder = [
            nn.Linear(dim_a, dim_a),
            nn.LayerNorm(dim_a),
            nn.SiLU(),
            nn.Linear(dim_a, dim_a),
            nn.LayerNorm(dim_a),
            nn.SiLU(),
        ]
        self.dual_modal_attn = nn.MultiHeadAttention(dim_a, num_heads=4, bias=True)
        self.ln1 = nn.LayerNorm(dim_a)
        self.user_agent_attn = nn.MultiHeadAttention(dim_a, num_heads=4, bias=True)
        self.ln2 = nn.LayerNorm(dim_a)
        self.to_cond_user = nn.Linear(dim_a, dim_h)

    def __call__(self, x: mx.array, c: mx.array, agent_audio: mx.array) -> mx.array:
        """x: user audio (kv), c: user motion (q), agent_audio: avatar audio (q2)."""
        for layer in self.motion_embedder:
            c = layer(c)
        cross_modal_feat = self.dual_modal_attn(c, x, x)
        cross_modal_feat = nn.silu(self.ln1(c + cross_modal_feat))
        unified = self.user_agent_attn(agent_audio, cross_modal_feat, cross_modal_feat)
        unified = nn.silu(self.ln2(agent_audio + unified))
        return self.to_cond_user(unified)


# =====================================================================
#  Attention (self-attention)
# =====================================================================

class Attention(nn.Module):
    """PyTorch: line 228.  Self-attention with RoPE, dense mask and a rolling KV cache."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 num_frames_for_clip: int = 50, num_frames_per_block: int = 10):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_frames_for_clip = num_frames_for_clip
        self.num_frames_per_block = num_frames_per_block

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, mask: mx.array, start_pos: int,
                 kv_cache: Optional[dict] = None, update_kv_cache: bool = False) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k_new, v_new = qkv[0], qkv[1], qkv[2]

        q = mx.fast.rope(q, self.head_dim, traditional=True, base=10000.0, scale=1.0, offset=start_pos)
        k_new = mx.fast.rope(k_new, self.head_dim, traditional=True, base=10000.0, scale=1.0, offset=start_pos)

        k_attn, v_attn = k_new, v_new
        if kv_cache is not None:
            if N == self.num_frames_per_block + 2:
                k_attn = mx.concatenate([kv_cache["k"], k_new], axis=2)
                v_attn = mx.concatenate([kv_cache["v"], v_new], axis=2)
                mask = mask[-(self.num_frames_per_block + 2):, :]
            if update_kv_cache:
                self.update_kv_cache(kv_cache, k_new, v_new)

        x = mx.fast.scaled_dot_product_attention(q, k_attn, v_attn, scale=self.scale, mask=mask)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

    def update_kv_cache(self, kv_cache: dict, k_new: mx.array, v_new: mx.array):
        """PyTorch: lines 305-316, as rebinds instead of in-place writes."""
        cache_size = kv_cache["k"].shape[2]
        if k_new.shape[2] >= cache_size:
            kv_cache["k"] = k_new[:, :, -cache_size - 2:-2, :]
            kv_cache["v"] = v_new[:, :, -cache_size - 2:-2, :]
        else:
            n = self.num_frames_per_block
            kv_cache["k"] = mx.concatenate([kv_cache["k"][:, :, n:, :], k_new[:, :, :-2, :]], axis=2)
            kv_cache["v"] = mx.concatenate([kv_cache["v"][:, :, n:, :], v_new[:, :, :-2, :]], axis=2)


# =====================================================================
#  CrossAttention
# =====================================================================

class CrossAttention(nn.Module):
    """PyTorch: line 138.  KV cache is 2 slots (attention_window)."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 num_frames_for_clip: int = 50, num_frames_per_block: int = 10):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_frames_per_block = num_frames_per_block

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, c: mx.array, mask: Optional[mx.array] = None,
                 kv_cache: Optional[dict] = None, update_kv_cache: bool = False) -> mx.array:
        B, N, C = x.shape
        Nc = c.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, Nc, 2, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        k_new, v_new = kv[0], kv[1]

        k_attn, v_attn = k_new, v_new
        if kv_cache is not None:
            if Nc == self.num_frames_per_block + 2:
                k_attn = mx.concatenate([kv_cache["k"], k_new], axis=2)
                v_attn = mx.concatenate([kv_cache["v"], v_new], axis=2)
                mask = cross_mask_local(q_len=N, cache_len=2)
            if update_kv_cache:
                self.append_kv_to_buffer(kv_cache, k_new, v_new)

        x = mx.fast.scaled_dot_product_attention(q, k_attn, v_attn, scale=self.scale, mask=mask)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(x)

    @staticmethod
    def append_kv_to_buffer(kv_cache: dict, k_new: mx.array, v_new: mx.array):
        """PyTorch: lines 220-225."""
        kv_cache["k"] = k_new[:, :, -4:-2, :]
        kv_cache["v"] = v_new[:, :, -4:-2, :]


# =====================================================================
#  DFoTBlock
# =====================================================================

class Mlp(nn.Module):
    """timm Mlp with tanh-approx GELU."""
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))


class DFoTBlock(nn.Module):
    """PyTorch: line 320.  adaLN-Zero conditioned block."""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 num_frames_for_clip: int = 50, num_frames_per_block: int = 10):
        super().__init__()
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              num_frames_for_clip=num_frames_for_clip,
                              num_frames_per_block=num_frames_per_block)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                         num_frames_for_clip=num_frames_for_clip,
                                         num_frames_per_block=num_frames_per_block)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))

    def __call__(self, x: mx.array, scale_shift_alpha, c: mx.array,
                 mask: mx.array, cross_mask: mx.array, start_pos: int,
                 kv_cache=None, update_kv_cache: bool = False) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = scale_shift_alpha
        self_kv_cache = kv_cache[0] if kv_cache is not None else None
        cross_kv_cache = kv_cache[1] if kv_cache is not None else None

        x_attn = self.attn(modulate(_layer_norm(x), shift_msa, scale_msa),
                           mask=mask, start_pos=start_pos,
                           kv_cache=self_kv_cache, update_kv_cache=update_kv_cache)
        x = x + gate_msa * x_attn
        x = x + self.cross_attn(_layer_norm(x), c=c, mask=cross_mask,
                                kv_cache=cross_kv_cache, update_kv_cache=update_kv_cache)
        x = x + gate_mlp * self.mlp(modulate(_layer_norm(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """PyTorch: line 667.  AdaLN + Linear."""
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        ]
        self.linear = nn.Linear(hidden_size, out_dim)

    def modulation(self, t_embed: mx.array) -> mx.array:
        return self.adaLN_modulation[1](nn.silu(t_embed))

    def __call__(self, x: mx.array, t_embed: mx.array) -> mx.array:
        shift, scale = mx.split(self.modulation(t_embed), 2, axis=-1)
        return self.linear(modulate(_layer_norm(x), shift, scale))

    def forward_precomputed(self, x: mx.array, final_adaLN) -> mx.array:
        shift, scale = final_adaLN
        return self.linear(modulate(_layer_norm(x), shift, scale))


# =====================================================================
#  FlowTransformer (top)
# =====================================================================

class FlowTransformerMLX(nn.Module):
    """PyTorch: line 378.  Inference-only (no condition dropout)."""
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_frames_for_clip = int(opt.sec * opt.fps)        # 50
        self.num_frames_per_block = opt.num_frames_per_block     # 10
        self.dtype = mx.float32

        self.x_embedder = SequenceEmbed(opt.dim_w * 2, opt.dim_h)
        self.t_embedder = TimestepEmbedder(opt.dim_h)
        self.c_embedder = AudioMotionAttention(opt.dim_a, opt.dim_h)
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(opt.dim_h, 6 * opt.dim_h, bias=True),
        ]
        self.blocks = [
            DFoTBlock(opt.dim_h, opt.num_heads, mlp_ratio=opt.mlp_ratio,
                      num_frames_for_clip=self.num_frames_for_clip,
                      num_frames_per_block=opt.num_frames_per_block)
            for _ in range(opt.transformer_depth)
        ]
        self.final_layer = FinalLayer(opt.dim_h, opt.dim_w)

        # Static masks (underscore prefix keeps them out of parameters()).
        self._alignment_mask = enc_dec_mask(
            self.num_frames_for_clip, self.num_frames_for_clip, 1, expansion=opt.attention_window)
        self._block_mask = make_block_bidir_causal_lookahead_mask(
            self.num_frames_for_clip, self.num_frames_per_block, 2)

    # ---------------- setup ----------------

    # Run once per block (or once per session) on a few dozen tokens, and their
    # outputs are cached, so they stay float32 regardless of `dtype`.
    FP32_MODULES = ("t_embedder", "c_embedder", "adaLN_modulation", "final_layer.adaLN_modulation")

    def set_dtype(self, dtype) -> None:
        """Cast the transformer stack's weights (and activations, via input
        casts) to `dtype`; the conditioning modules stay float32."""
        self.dtype = dtype
        super().set_dtype(dtype)
        for name in self.FP32_MODULES:
            mod = self
            for part in name.split("."):
                mod = mod[int(part)] if isinstance(mod, list) else getattr(mod, part)
            if isinstance(mod, list):
                for m in mod:
                    m.set_dtype(mx.float32)
            else:
                mod.set_dtype(mx.float32)

    def init_kv_cache(self, batch_size: int) -> list:
        """Zero KV caches, shape (B, H, L, D).  batch_size includes the CFG
        branches (branches * B).  Mirrors AvatarForcing.initialize_kv_cache."""
        cache_len = self.num_frames_for_clip - self.num_frames_per_block - 2   # 38
        cross_len = self.opt.attention_window                                  # 2
        H = self.opt.num_heads
        D = self.opt.dim_h // H
        z = lambda L: mx.zeros((batch_size, H, L, D), dtype=self.dtype)
        return [({"k": z(cache_len), "v": z(cache_len)}, {"k": z(cross_len), "v": z(cross_len)})
                for _ in self.blocks]

    # ---------------- precompute ----------------

    def adaLN(self, t_embed: mx.array) -> mx.array:
        return self.adaLN_modulation[1](nn.silu(t_embed))

    def compute_condition(self, wa: mx.array, wa_user: mx.array, motion_user: mx.array) -> mx.array:
        """PyTorch: lines 527-531 (note the argument order into c_embedder)."""
        f32 = lambda a: a.astype(mx.float32)
        return self.c_embedder(f32(wa_user), f32(motion_user), f32(wa)).astype(self.dtype)

    def compute_wr_embedded(self, wr: mx.array, seq_len: int) -> mx.array:
        """PyTorch: lines 534-537.  wr: (B, dim_w) -> (B, seq_len, dim_w)."""
        return mx.broadcast_to(wr[:, None, :], (wr.shape[0], seq_len, wr.shape[-1])).astype(self.dtype)

    def precompute_timestep_adaLN(self, timestep_list, batch_size: int, seq_len: int,
                                  context_len: int = 0) -> list:
        """PyTorch: lines 580-623.  Returns one (scale_shift_alpha[6], t_embed,
        final_adaLN[2]) tuple per timestep; every array is (batch_size, L, ·)
        with L = context_len + seq_len (context tokens use t=0)."""
        ts = list(timestep_list) + ([0.0] if context_len > 0 else [])
        t_embed = self.t_embedder(mx.array(ts, dtype=mx.float32))
        ada = self.adaLN(t_embed)
        final_ada = self.final_layer.modulation(t_embed)

        L = context_len + seq_len
        out = []
        for i in range(len(timestep_list)):
            def per_token(a):
                rows = a[i:i + 1]
                if context_len > 0:
                    a_tok = mx.concatenate([mx.repeat(a[-1:], context_len, axis=0),
                                            mx.repeat(rows, seq_len, axis=0)], axis=0)
                else:
                    a_tok = mx.repeat(rows, seq_len, axis=0)
                return mx.broadcast_to(a_tok[None], (batch_size, L, a.shape[-1])).astype(self.dtype)
            out.append((tuple(mx.split(per_token(ada), 6, axis=-1)),
                        per_token(t_embed),
                        tuple(mx.split(per_token(final_ada), 2, axis=-1))))
        return out

    def precompute_zero_adaLN(self, batch_size: int, seq_len: int):
        """adaLN for t=0 on every token — used by the KV-cache update pass
        (AvatarForcing.update_kv_cache)."""
        return self.precompute_timestep_adaLN([0.0], batch_size, seq_len, 0)[0]

    # ---------------- forward ----------------

    def forward_with_precomputed(self, x: mx.array, precomputed_c: mx.array, precomputed_adaLN,
                                 precomputed_wr: mx.array, start_pos: int,
                                 kv_cache_list: Optional[list] = None,
                                 update_kv_cache: bool = False) -> mx.array:
        """PyTorch: lines 500-518.  Returns float32 (B, N, dim_w)."""
        shared_scale_shift_alpha, _t_embed, final_adaLN = precomputed_adaLN
        x = mx.concatenate([x.astype(self.dtype), precomputed_wr], axis=-1)
        x = self.x_embedder(x)
        for i, block in enumerate(self.blocks):
            x = block(x, shared_scale_shift_alpha, precomputed_c,
                      mask=self._block_mask, cross_mask=self._alignment_mask,
                      start_pos=start_pos,
                      kv_cache=kv_cache_list[i] if kv_cache_list is not None else None,
                      update_kv_cache=update_kv_cache)
        return self.final_layer.forward_precomputed(x, final_adaLN).astype(mx.float32)

    def __call__(self, t: mx.array, x: mx.array, wa: mx.array, wa_user: mx.array,
                 motion_user: mx.array, wr: mx.array, start_pos: int = 0,
                 kv_cache_list=None, update_kv_cache: bool = False) -> mx.array:
        """Full forward (PyTorch `forward`, train=False).  t: (B, L) per-token timesteps."""
        B, L = t.shape
        t_embed = self.t_embedder(t.reshape(-1)).reshape(B, L, -1)
        cast = lambda parts: tuple(p.astype(self.dtype) for p in parts)
        adaLN = (cast(mx.split(self.adaLN(t_embed), 6, axis=-1)), t_embed.astype(self.dtype),
                 cast(mx.split(self.final_layer.modulation(t_embed), 2, axis=-1)))
        c = self.compute_condition(wa, wa_user, motion_user)
        wr_emb = self.compute_wr_embedded(wr, x.shape[1])
        return self.forward_with_precomputed(x, c, adaLN, wr_emb, start_pos,
                                             kv_cache_list, update_kv_cache)


# =====================================================================
#  Weight loading
# =====================================================================

def load_weights(model: FlowTransformerMLX, safetensors_path: str, dtype=mx.float32) -> None:
    """Load converted weights (see convert.py), strictly, and cast to `dtype`."""
    model.load_weights(safetensors_path, strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
