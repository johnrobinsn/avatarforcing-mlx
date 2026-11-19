"""RoPE and attention masks vs the original helpers (no weights needed)."""
import numpy as np
import pytest
import mlx.core as mx

from avatarforcing_mlx import helpers as hm

from conftest import _require_upstream, device_for, to_np, t


@pytest.fixture(scope="module")
def tft():
    _require_upstream()
    import models.avatarforcing.flow_transformer as tft
    return tft


def test_precompute_freqs_cis(tft):
    ref, out = tft.precompute_freqs_cis(128, 2048), hm.precompute_freqs_cis(128, 2048)
    assert np.max(np.abs(ref.real.numpy() - to_np(mx.real(out)))) < 1e-3
    assert np.max(np.abs(ref.imag.numpy() - to_np(mx.imag(out)))) < 1e-3


@pytest.mark.parametrize("start", [0, 38, 1000])
def test_rotary_matches(tft, start):
    x = np.random.default_rng(1).standard_normal((2, 8, 12, 128)).astype(np.float32)
    fc = tft.precompute_freqs_cis(128, 1024 + 12)
    ref = tft.apply_rotary_emb_q(t(x), fc, start_pos=start).numpy()
    table = hm.apply_rotary_emb_q(mx.array(x), hm.precompute_freqs_cis(128, 1024 + 12)[start:start + 12])
    with device_for("strict"):
        fast = mx.fast.rope(mx.array(x), 128, traditional=True, base=10000.0, scale=1.0, offset=start)
        fast = to_np(fast)
    assert np.max(np.abs(ref - to_np(table))) < 1e-3
    assert np.max(np.abs(ref - fast)) < 1e-3          # what the model uses


def test_masks(tft):
    eq = lambda a, b: np.array_equal(to_np(a), to_np(b))
    assert eq(tft.enc_dec_mask(50, 50, 1, 2), hm.enc_dec_mask(50, 50, 1, 2))
    assert eq(tft.make_block_bidir_causal_lookahead_mask(50, 10, 2), hm.make_block_bidir_causal_lookahead_mask(50, 10, 2))
    assert eq(tft.CrossAttention.cross_mask_local(12, 2), hm.cross_mask_local(12, 2))
    for args in [(0, 0, 50, 10, 2), (38, 38, 12, 10, 2), (128, 38, 12, 10, 2)]:
        assert eq(tft.make_block_bidir_causal_lookahead_mask_window(*args),
                  hm.make_block_bidir_causal_lookahead_mask_window(*args))
    for args in [(0, 0, 50, 2), (38, 2, 12, 2)]:
        assert eq(tft.enc_dec_mask_window(*args), hm.enc_dec_mask_window(*args))
