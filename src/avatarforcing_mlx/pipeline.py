"""AvatarForcing on Apple Silicon: MLX flow transformer + torch-MPS encoders/decoder.

    avatar portrait ─► motion encoder (torch MPS, fp32, once)   ─► s_r, feats, r_s
    avatar audio    ─► wav2vec2 + audio projection (torch MPS)  ─► wa           [1, T, 512]
    user audio      ─► wav2vec2 + audio projection (optional)   ─► user_wa      [1, T, 512]
    user video      ─► motion encoder (optional)                ─► user_motion  [1, T, 512]
    conditions      ─► FlowSamplerMLX (MLX, fp32 by default)    ─► motion latents
    latents         ─► motion decoder (torch MPS, fp16, decoder_fast + torch.compile) ─► RGB

`stream()` follows AvatarForcing.sample / inference_stream: a 50-frame primer, then
10-frame blocks with 2 frames of overlap, yielding decoded frames per block.

Precision: the flow transformer defaults to float32.  bf16 is faster, but diffusion
sampling amplifies its rounding (as it does for PyTorch's own bf16); see docs/PERF.md.

MLX streams are thread-local: create and use a pipeline on the same thread.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import torch
import mlx.core as mx

from . import decoder_fast, weights
from .flow_transformer import FlowTransformerMLX, load_weights
from .media import FPS, SAMPLE_RATE, normalize_audio
from .sampler import FlowSamplerMLX
from .torch_models.audio_encoder import AudioEncoder
from .torch_models.avatarforcing.generator import Generator


def load_opt(wav2vec_path=None):
    from omegaconf import OmegaConf
    opt = OmegaConf.load(str(resources.files("avatarforcing_mlx") / "configs" / "inference.yaml"))
    if wav2vec_path is not None:
        opt.wav2vec_model_path = str(wav2vec_path)
    return opt


def _load_params(module: torch.nn.Module, ckpt, prefix: str) -> torch.nn.Module:
    """Copy parameters only, like AvatarForcing's inference.py load_ckpt (buffers are
    not loaded — the checkpoint's mask buffers are inverted, see weights.RENAME_MAP)."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    params = dict(module.named_parameters())
    missing = set(params) - set(sd)
    if missing:
        raise KeyError(f"{ckpt}: missing parameters {sorted(missing)[:5]}")
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(sd[k])
    return module


def load_generator(path, device) -> Generator:
    g = Generator(512, 512, 20)
    return _load_params(g, path, "motion_autoencoder.").to(device).eval()


def load_audio_encoder(opt, flow_transformer_pth, device) -> AudioEncoder:
    # transformers warns that wav2vec2 `pos_conv_embed.conv.weight_g/weight_v` were
    # "not used"; that's spurious — torch's weight_norm parametrization remaps those
    # legacy keys on load (effective conv weight == weight_g * v / ||v||).
    enc = AudioEncoder(opt)
    sd = torch.load(flow_transformer_pth, map_location="cpu", weights_only=True)
    p = "audio_encoder.audio_projection."
    enc.audio_projection.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)})
    return enc.to(device).eval()


def image_to_tensor(image: np.ndarray, device) -> torch.Tensor:
    """RGB uint8 [512, 512, 3] -> [1, 3, 512, 512] in [-1, 1] (A.Normalize(0.5, 0.5))."""
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return (x * 2 - 1).unsqueeze(0).to(device)


@dataclass
class BlockTiming:
    flow_ms: float
    decode_ms: float
    total_ms: float


@dataclass
class StreamStats:
    timings: list = field(default_factory=list)
    ewma_ms: float | None = None
    alpha: float = 0.1

    def add(self, t: BlockTiming):
        self.timings.append(t)
        self.ewma_ms = t.total_ms if self.ewma_ms is None else \
            (1 - self.alpha) * self.ewma_ms + self.alpha * t.total_ms


def _pad(a: mx.array, L: int) -> mx.array:
    """Replicate-pad along time to length L (F.pad(..., mode='replicate') in AvatarForcing)."""
    return a if a.shape[1] >= L else mx.concatenate([a, mx.repeat(a[:, -1:], L - a.shape[1], axis=1)], axis=1)


class AvatarPipeline:
    def __init__(self, weights_dir=None, flow_dtype: str = "float32", decoder_dtype: str = "float16",
                 compile_decoder: bool = True, nfe: int = 10, a_cfg_scale: float = 2.0,
                 u_cfg_scale: float = 1.0, device: str = "mps"):
        p = weights.paths(weights_dir)
        if missing := weights.missing(weights_dir):
            raise FileNotFoundError(f"model files missing in {p['root']}: {missing} "
                                    f"(run `avatarforcing-mlx download`)")
        self.opt = opt = load_opt(p["wav2vec2"])
        self.device = device
        self.clip = int(opt.sec * opt.fps)          # 50
        self.block = opt.num_frames_per_block       # 10

        self.flow = FlowTransformerMLX(opt)
        load_weights(self.flow, str(p["mlx"]), dtype=getattr(mx, flow_dtype))
        self.sampler = FlowSamplerMLX(self.flow, nfe=nfe, a_cfg_scale=a_cfg_scale, u_cfg_scale=u_cfg_scale)

        self.generator = load_generator(p["motion_autoencoder"], device)
        self.audio_encoder = load_audio_encoder(opt, p["flow_transformer"], device)

        # Encoder and `direction` stay fp32 (torch.linalg.qr has no fp16 MPS kernel);
        # the decoder's synthesis path runs in `decoder_dtype` with the fast kernels.
        decoder_fast.patch()
        self.decoder_dtype = getattr(torch, decoder_dtype)
        self.generator.dec.to(self.decoder_dtype)
        self.generator.dec.direction.float()
        self._decode = self.generator.dec
        self._compile = compile_decoder
        self._compiled = None
        self.s_r = None

    # ---------------- encoders ----------------

    @torch.inference_mode()
    def set_reference(self, image: np.ndarray):
        """Encode the avatar portrait once.  image: RGB uint8 [512, 512, 3] (face-cropped)."""
        enc = self.generator.enc
        s_r, _, feats = enc(image_to_tensor(image, self.device), input_target=None)
        r_s = self.generator.dec.direction(enc.fc(s_r))
        self.s_r = s_r.to(self.decoder_dtype).unsqueeze(1)                     # [1, 1, 512]
        self.feats = [f.to(self.decoder_dtype).repeat_interleave(self.block, 0) for f in feats]
        self.r_s = mx.array(r_s.float().cpu().numpy())                         # [1, 512]
        self._warm_decoder()

    @torch.inference_mode()
    def audio_features(self, pcm_16k: np.ndarray, n_frames: int | None = None,
                       normalize: bool = True) -> mx.array:
        """wav2vec2 + audio projection -> MLX [1, T, 512].  normalize=True matches
        AvatarForcing's DataProcessor (Wav2Vec2FeatureExtractor)."""
        if n_frames is None:
            n_frames = int(np.ceil(len(pcm_16k) * FPS / SAMPLE_RATE))
        pcm = normalize_audio(pcm_16k) if normalize else np.asarray(pcm_16k, dtype=np.float32)
        a = torch.from_numpy(pcm).unsqueeze(0).to(self.device)
        return mx.array(self.audio_encoder.inference(a, seq_len=n_frames).float().cpu().numpy())

    @torch.inference_mode()
    def user_motion(self, frames: np.ndarray) -> mx.array:
        """AvatarForcing.encode_user_motion: face-cropped user frames RGB uint8
        [T, 512, 512, 3] -> MLX [1, T, 512], encoded in chunks of 50 frames."""
        enc, direction = self.generator.enc, self.generator.dec.direction
        out = []
        for i in range(0, len(frames), self.clip):
            x = torch.cat([image_to_tensor(f, self.device) for f in frames[i:i + self.clip]], dim=0)
            out.append(direction(enc.enc_motion(x)).float().cpu())
        return mx.array(torch.cat(out, dim=0).unsqueeze(0).numpy())

    # ---------------- decoder ----------------

    def _warm_decoder(self):
        """First (eager) call fills decoder_fast's grid cache; then compile."""
        z = torch.zeros(1, self.block, 512, device=self.device)
        self.decode(z)
        if self._compile and self._compiled is None:
            self._compiled = torch.compile(self.generator.dec, dynamic=False)
            self._decode = self._compiled
            self.decode(z)

    @torch.inference_mode()
    def decode(self, latents) -> torch.Tensor:
        """latents [1, k, 512] (MLX or torch, k <= 10) -> pixels [k, 3, 512, 512] in [-1, 1]."""
        if isinstance(latents, mx.array):
            latents = torch.from_numpy(np.array(latents.astype(mx.float32)))
        latents = latents.to(self.device, self.decoder_dtype)
        k = latents.shape[1]
        if k < self.block:                       # replicate-pad to the compiled block size
            latents = torch.cat([latents, latents[:, -1:].expand(1, self.block - k, -1)], dim=1)
        img, _ = self._decode((self.s_r + latents).reshape(self.block, -1), alpha=None, feats=self.feats)
        return img[:k]

    # ---------------- streaming ----------------

    def stream(self, wa: mx.array, user_wa: mx.array | None = None, user_motion: mx.array | None = None,
               seed: int = 25, noises: list | None = None, emit_primer: bool = True,
               stats: StreamStats | None = None):
        """Yield (block_idx, pixels [k, 3, H, W] torch, BlockTiming) for T = wa.shape[1] frames.

        user_wa / user_motion: optional [1, >=T, 512] conditioning on the conversation
        partner (AvatarForcing's interactive mode, weighted by u_cfg_scale).
        noises: optional list of numpy arrays, [1, 50, 512] then [1, 10, 512] per block."""
        if self.s_r is None:
            raise RuntimeError("call set_reference() first")
        T = wa.shape[1]
        with_user = user_wa is not None or user_motion is not None
        rng = np.random.default_rng(seed)
        noise = lambda i, L: mx.array(noises[i] if noises is not None else
                                      rng.standard_normal((1, L, 512)).astype(np.float32))
        def win(a, lo, L):   # frames [lo, lo + L), replicate-padded past the end
            if a is None:
                return None
            return _pad(a[:, lo:lo + L] if lo < a.shape[1] else a[:, -1:], L)

        s = self.sampler
        s.reset(1, with_user=with_user)
        t0 = time.perf_counter()
        x_t = s.prime(win(wa, 0, self.clip), self.r_s, noise(0, self.clip),
                      win(user_wa, 0, self.clip), win(user_motion, 0, self.clip))
        mx.eval(x_t, s.kv_cache)
        t_flow = time.perf_counter()
        if emit_primer:
            for b in range(0, min(self.clip, T), self.block):
                t1 = time.perf_counter()
                pix = self.decode(x_t[:, b:min(b + self.block, T)])
                torch.mps.synchronize()
                yield b // self.block, pix, BlockTiming((t_flow - t0) * 1000 if b == 0 else 0.0,
                                                        (time.perf_counter() - t1) * 1000, 0.0)

        L = self.block + 2
        for i, t_idx in enumerate(range(self.clip, T, self.block)):
            t0 = time.perf_counter()
            lo = t_idx - 2
            x_t = s.step(win(wa, lo, L), self.r_s, x_t, noise(i + 1, self.block), lo,
                         win(user_wa, lo, L), win(user_motion, lo, L))
            latents = x_t[:, -self.block:][:, :min(self.block, T - t_idx)]
            mx.eval(latents, s.kv_cache)
            t1 = time.perf_counter()
            pix = self.decode(latents)
            torch.mps.synchronize()
            t2 = time.perf_counter()
            timing = BlockTiming((t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000)
            if stats is not None:
                stats.add(timing)
            yield t_idx // self.block, pix, timing
