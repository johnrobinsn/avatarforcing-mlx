"""Shared fixtures: the original PyTorch AvatarForcing (reference) and the MLX port.

The reference is the authors' code at the pinned commit in third_party/AvatarForcing
(run scripts/fetch_upstream.sh), loaded from the original checkpoints; tests are
skipped if either is missing.

Two MLX modes:
  strict — float32 on the MLX CPU stream: checks the port is mathematically the same
           (fp32 matmuls on Apple GPUs carry ~1e-3 relative error, which would hide bugs)
  bf16   — bfloat16 weights/activations on the GPU; judged against PyTorch's own bf16
           (CPU autocast) error, since bf16 rounding alone moves outputs by ~1%.
"""
from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import mlx.core as mx

from avatarforcing_mlx import weights
from avatarforcing_mlx.flow_transformer import FlowTransformerMLX, load_weights
from avatarforcing_mlx.pipeline import load_opt

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "AvatarForcing"
if UPSTREAM.exists():
    sys.path.insert(0, str(UPSTREAM))

PATHS = weights.paths()


def _require_upstream():
    if not (UPSTREAM / "models" / "avatarforcing" / "flow_transformer.py").exists():
        pytest.skip("original AvatarForcing code not found — run scripts/fetch_upstream.sh")


def _require_weights():
    if weights.missing():
        pytest.skip(f"model files missing in {PATHS['root']} — run `avatarforcing-mlx download`")


# ---------------- numerics ----------------

def to_np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().float().cpu().numpy()
    if isinstance(a, mx.array):
        return np.array(a.astype(mx.float32))
    return np.asarray(a)


def t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a))


def rel_rms(ref, out) -> float:
    r, o = to_np(ref), to_np(out)
    return float(np.sqrt(np.mean((r - o) ** 2)) / max(np.sqrt(np.mean(r ** 2)), 1e-12))


def assert_close(ref, out, mode: str, tol: float, baseline=None, name: str = ""):
    """strict: max |ref - out| <= tol * max(1, max|ref|).
    bf16: relative RMS error <= tol, or <= 1.25x PyTorch-bf16's own error (`baseline`)."""
    r, o = to_np(ref), to_np(out)
    assert r.shape == o.shape, f"{name}: shape {r.shape} vs {o.shape}"
    if mode == "strict":
        d, scale = float(np.max(np.abs(r - o))), float(np.max(np.abs(r)))
        assert d <= tol * max(1.0, scale), f"{name}: max_abs {d:.2e} (scale {scale:.2e}, tol {tol:.0e})"
    else:
        err = rel_rms(r, o)
        limit = tol if baseline is None else max(tol, 1.25 * rel_rms(r, baseline))
        assert err <= limit, f"{name}: rel_rms {err:.2e} > {limit:.2e}"


def audio_feats(rng, B, L, dim=512) -> np.ndarray:
    """Shaped like AudioEncoder output (Linear -> LayerNorm -> SiLU)."""
    x = rng.standard_normal((B, L, dim)).astype(np.float32)
    return (x / (1 + np.exp(-x))).astype(np.float32)


@contextlib.contextmanager
def device_for(mode: str):
    prev = mx.default_device()
    mx.set_default_device(mx.cpu if mode == "strict" else mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(prev)


def torch_bf16():
    return torch.autocast("cpu", dtype=torch.bfloat16)


# ---------------- models ----------------

@pytest.fixture(scope="session")
def opt():
    o = load_opt(PATHS["wav2vec2"])
    o.rank = "cpu"
    return o


@pytest.fixture(scope="session")
def torch_ft(opt):
    """The original FlowTransformer with the original checkpoint (parameters only,
    as inference.py:load_ckpt does — the checkpoint's mask buffers are inverted)."""
    _require_upstream()
    _require_weights()
    import models.avatarforcing.flow_transformer as tft
    torch.manual_seed(0)
    m = tft.FlowTransformer(opt)
    sd = torch.load(PATHS["flow_transformer"], map_location="cpu", weights_only=True)
    sd = {k[len("flow_transformer."):]: v for k, v in sd.items() if k.startswith("flow_transformer.")}
    params = dict(m.named_parameters())
    assert set(params) == set(sd) - {"alignment_mask", "block_bidir_causal_lookahead_mask"}
    with torch.no_grad():
        for k, p in params.items():
            p.copy_(sd[k])
    return m.eval()


_MLX = {}


@pytest.fixture(scope="session")
def mlx_ft(opt):
    _require_weights()

    def get(mode: str) -> FlowTransformerMLX:
        if mode not in _MLX:
            with device_for(mode):
                m = FlowTransformerMLX(opt)
                load_weights(m, str(PATHS["mlx"]), dtype=mx.float32 if mode == "strict" else mx.bfloat16)
            _MLX[mode] = m
        return _MLX[mode]
    return get


@pytest.fixture(scope="session")
def upstream_af(torch_ft, opt):
    """Factory for an object carrying AvatarForcing's own sampling methods
    (solve_cfg, solve_closed, prepare_cfg_condition, update_kv_cache,
    initialize_kv_cache), bound to the reference flow transformer."""
    from models import FlowMatchScheduler
    from models.avatarforcing.AvatarForcing import AvatarForcing

    def make(nfe: int = 10):
        s = types.SimpleNamespace(opt=opt, rank="cpu", flow_transformer=torch_ft)
        s.num_frames_for_clip = int(opt.sec * opt.fps)
        s.num_frames_per_block = opt.num_frames_per_block
        s.scheduler = FlowMatchScheduler(num_steps=1000, shift=1, sigma_min=0.0, extra_one_step=False)
        s.scheduler.set_timesteps(opt.num_train_timestep, training=True)
        s.denoising_step_list = torch.tensor(np.linspace(opt.num_train_timestep, 0, nfe - 1).tolist())
        for name in ("solve_cfg", "solve_closed", "prepare_cfg_condition", "update_kv_cache", "initialize_kv_cache"):
            setattr(s, name, types.MethodType(getattr(AvatarForcing, name), s))
        return s
    return make
