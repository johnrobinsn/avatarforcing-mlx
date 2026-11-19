"""Vendored PyTorch files are the upstream files, changed only where their headers say
(provenance check), and the decoder fast kernels are exact."""
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import UPSTREAM, PATHS, _require_upstream, _require_weights

PKG = Path(__file__).resolve().parents[1] / "src" / "avatarforcing_mlx"
VENDORED = {   # package file: (upstream file, {vendored line: upstream line})
    "torch_models/base.py": ("models/__init__.py", {}),
    "torch_models/wav2vec2.py": ("models/wav2vec2.py", {}),
    "torch_models/avatarforcing/encoder.py": ("models/avatarforcing/encoder.py", {}),
    "torch_models/avatarforcing/styledecoder.py": ("models/avatarforcing/styledecoder.py", {}),
    "torch_models/avatarforcing/generator.py": ("models/avatarforcing/generator.py",
                                                {"from ..base import BaseModel": "from models import BaseModel"}),
}


def _strip_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "".join(lines[i + 1:])        # header + one blank line


@pytest.mark.parametrize("pkg_file", sorted(VENDORED))
def test_vendored_file_matches_upstream(pkg_file):
    _require_upstream()
    up_file, subs = VENDORED[pkg_file]
    ours = _strip_header((PKG / pkg_file).read_text())
    for a, b in subs.items():
        ours = ours.replace(a, b)
    assert ours == (UPSTREAM / up_file).read_text()


def test_audio_encoder_excerpt_matches_upstream():
    _require_upstream()
    up = (UPSTREAM / "models/avatarforcing/AvatarForcing.py").read_text()
    excerpt = up[up.index("class AudioEncoder(BaseModel):"):up.index("######## Main Model ########")].rstrip()
    ours = (PKG / "torch_models/audio_encoder.py").read_text()
    assert excerpt in ours


def test_decoder_fast_kernels_exact():
    """decoder_fast's rewrites are exact algebra: in float64 (CPU, real weights) they
    reproduce the original decoder to rounding level.  (In fp32 the reordered sums
    differ by ~1e-5 on real portraits.)"""
    _require_weights()
    from avatarforcing_mlx import decoder_fast
    from avatarforcing_mlx.pipeline import load_generator
    decoder_fast.unpatch()
    g = load_generator(PATHS["motion_autoencoder"], "cpu").double()
    torch.manual_seed(0)
    img = (torch.rand(1, 3, 512, 512) * 2 - 1).double()
    with torch.inference_mode():
        s_r, _, feats = g.enc(img, input_target=None)
        s = s_r + 0.1 * torch.randn(2, 512, dtype=torch.float64)
        fe = [f.repeat_interleave(2, 0) for f in feats]
        ref, _ = g.dec(s, alpha=None, feats=fe)
        decoder_fast.patch()
        try:
            out, _ = g.dec(s, alpha=None, feats=fe)
        finally:
            decoder_fast.unpatch()
    assert float((ref - out).abs().max()) < 1e-9
