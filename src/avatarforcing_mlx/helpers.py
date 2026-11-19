# MLX port of AvatarForcing (https://github.com/TaekyungKi/AvatarForcing), an adaptation of
# models/avatarforcing/flow_transformer.py at commit 507e3046ad53. Copyright the AvatarForcing authors; MLX port by
# John Robinson. Licensed CC BY-NC 4.0 (see LICENSE and NOTICE).

"""MLX counterparts for the small helpers in AvatarForcing's flow_transformer.py:
  - RoPE precompute + apply
  - Block-bidir-causal lookahead masks
  - adaLN modulate

Keep these simple and well-tested — everything else depends on them.

Note: the model's hot path uses `mx.fast.rope(traditional=True, offset=start_pos)`
rather than the table-based helpers below.  It computes the same interleaved-pair
rotation as PyTorch's view_as_complex path, without a precomputed table (so no
max_seq_len limit).  The table-based helpers are kept for equivalence testing.
"""
from __future__ import annotations

import mlx.core as mx


# ---------- Rotary embeddings ----------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> mx.array:
    """Polar-form complex rotary frequencies.

    Returns: (end, dim/2) complex64 array.

    PyTorch reference (flow_transformer.py:41-47):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # [end, dim/2]
    """
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even.")
    idx = mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32)
    inv_freq = 1.0 / (theta ** (idx / dim))
    t = mx.arange(end).astype(mx.float32)
    freqs = mx.outer(t, inv_freq)                     # (end, dim/2)
    return mx.cos(freqs) + 1j * mx.sin(freqs)          # complex64


def _apply_rotary(x: mx.array, freqs_cis_n: mx.array) -> mx.array:
    """Rotate interleaved (even, odd) pairs of the last axis by freqs_cis_n.

    Real-valued form of PyTorch's view_as_complex(x) * freqs_cis.
      x: (B, H, N, D);  freqs_cis_n: (N, D/2) complex64
    """
    B, H, N, D = x.shape
    cos = mx.real(freqs_cis_n).reshape(1, 1, N, D // 2)
    sin = mx.imag(freqs_cis_n).reshape(1, 1, N, D // 2)
    pair = x.astype(mx.float32).reshape(B, H, N, D // 2, 2)
    xr, xi = pair[..., 0], pair[..., 1]
    out = mx.stack([xr * cos - xi * sin, xr * sin + xi * cos], axis=-1)
    return out.reshape(B, H, N, D).astype(x.dtype)


def apply_rotary_emb_q(xq: mx.array, freqs_cis_n: mx.array) -> mx.array:
    """Apply rotary to Q.  PyTorch reference: flow_transformer.py:76-84."""
    return _apply_rotary(xq, freqs_cis_n)


def apply_rotary_emb_k(xk: mx.array, freqs_cis_n: mx.array) -> mx.array:
    """Apply rotary to K.  PyTorch reference: flow_transformer.py:86-93."""
    return _apply_rotary(xk, freqs_cis_n)


# ---------- Masks ----------

def make_block_bidir_causal_lookahead_mask(T: int, local_len: int, lookahead: int = 2) -> mx.array:
    """Static (T, T) self-attention mask.  PyTorch: flow_transformer.py:25-31.

    This is the mask actually used by FlowTransformer: the full (50, 50) matrix
    at warmup, and its last (num_frames_per_block + 2) rows at steady state.
    """
    q = mx.arange(T)[:, None]
    k = mx.arange(T)[None, :]
    return ((k // local_len) < (q // local_len)) | (k <= q + lookahead)


def make_block_bidir_causal_lookahead_mask_window(
    start_pos: int,
    cache_len: int,
    n_tokens: int,
    local_len: int,
    lookahead: int = 2,
) -> mx.array:
    """Windowed self-attention mask.  PyTorch: flow_transformer.py:33-39.

    Unused by the PyTorch inference path (kept for parity).
    """
    q_abs = mx.arange(start_pos, start_pos + n_tokens)[:, None]
    k_abs = mx.arange(start_pos - cache_len, start_pos + n_tokens)[None, :]
    q_blk = mx.floor_divide(q_abs, local_len)
    k_blk = mx.floor_divide(k_abs, local_len)
    return (k_blk < q_blk) | (k_abs <= q_abs + lookahead)


def enc_dec_mask_window(
    start_pos: int,
    cache_len: int,
    n_tokens: int,
    expansion: int,
) -> mx.array:
    """Windowed cross-attention mask.  PyTorch: flow_transformer.py:19-23."""
    q_abs = mx.arange(start_pos, start_pos + n_tokens)[:, None]
    k_abs = mx.arange(start_pos - cache_len, start_pos + n_tokens)[None, :]
    return (k_abs >= q_abs - expansion) & (k_abs <= q_abs + expansion)


def enc_dec_mask(T: int, S: int, frame_width: int = 1, expansion: int = 2) -> mx.array:
    """Alignment mask used at warmup (non-streaming path).

    PyTorch reference (flow_transformer.py:13-17):
        mask = torch.ones(T, S)
        for i in range(T):
            mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
        return mask == 0
    """
    q = mx.arange(T)[:, None]
    k = mx.arange(S)[None, :]
    lo = mx.maximum(0, (q - expansion) * frame_width)
    hi = (q + expansion + 1) * frame_width
    return (k >= lo) & (k < hi)


def cross_mask_local(q_len: int = 10, cache_len: int = 2, expansion: int = 2) -> mx.array:
    """Steady-state cross-attention mask.  PyTorch: CrossAttention.cross_mask_local."""
    q = mx.arange(q_len)[:, None]
    k = mx.arange(cache_len + q_len)[None, :]
    return mx.abs(k - (q + cache_len)) <= expansion


# ---------- Modulation ----------

def modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    """x * (1 + scale) + shift.  PyTorch: flow_transformer.py:10-11."""
    return x * (1.0 + scale) + shift
