"""Attention, cross-attention, condition embedder, adaLN precompute and the full
forward pass vs the original PyTorch modules (real weights)."""
import numpy as np
import pytest
import torch
import mlx.core as mx

from conftest import assert_close, audio_feats, device_for, t, torch_bf16

MODES = [("strict", 1e-4), ("bf16", 1e-2)]
FWD_MODES = [("strict", 1e-3), ("bf16", 2e-2)]


def _kv(rng, B, L, H=8, D=128):
    return {"k": rng.standard_normal((B, H, L, D)).astype(np.float32),
            "v": rng.standard_normal((B, H, L, D)).astype(np.float32)}


@pytest.fixture(scope="module")
def module_case(torch_ft):
    T = torch_ft
    r = np.random.default_rng(2)
    B, H = 6, 1024                               # 6 = 3 CFG branches x 2
    case = dict(x50=r.standard_normal((B, 50, H)).astype(np.float32),
                x12=r.standard_normal((B, 12, H)).astype(np.float32),
                c12=r.standard_normal((B, 12, H)).astype(np.float32),
                kv_self=_kv(r, B, 38), kv_cross=_kv(r, B, 2), start=438,
                wa=audio_feats(r, B, 12), ts=np.linspace(1000, 0, 9).tolist())
    case["user"] = audio_feats(r, B, 12)
    case["motion"] = r.standard_normal((B, 12, 512)).astype(np.float32) * 0.3

    def refs():
        ta, tc = T.blocks[3].attn, T.blocks[3].cross_attn
        bm, am, c = T.block_bidir_causal_lookahead_mask, T.alignment_mask, case
        R = {}
        with torch.no_grad():
            R["attn50"] = ta(t(c["x50"]), mask=bm, start_pos=0)
            R["kv"] = {k: t(v.copy()) for k, v in c["kv_self"].items()}
            R["attn12"] = ta(t(c["x12"]), mask=bm, start_pos=c["start"], kv_cache=R["kv"], update_kv_cache=True)
            R["kv50"] = {k: t(v.copy()) for k, v in c["kv_self"].items()}
            ta(t(c["x50"]), mask=bm, start_pos=0, kv_cache=R["kv50"], update_kv_cache=True)
            R["cross50"] = tc(t(c["x50"]), c=t(c["x50"]), mask=am)
            R["ckv"] = {k: t(v.copy()) for k, v in c["kv_cross"].items()}
            R["cross12"] = tc(t(c["x12"]), c=t(c["c12"]), mask=am, kv_cache=R["ckv"], update_kv_cache=True)
            z = np.zeros_like(c["wa"])
            R["cond_null"] = T.compute_condition(t(c["wa"]), t(z), t(z))
            R["cond_user"] = T.compute_condition(t(c["wa"]), t(c["user"]), t(c["motion"]))
            R["ada"] = T.precompute_timestep_adaLN(c["ts"], B, 10, "cpu", context_len=2)
        return R
    R = refs()
    with torch_bf16():
        BASE = refs()
    return case, R, BASE


@pytest.mark.parametrize("mode,tol", MODES)
def test_attention_and_cross_attention(mlx_ft, module_case, mode, tol):
    c, R, BASE = module_case
    with device_for(mode):
        M = mlx_ft(mode)
        ma, mc = M.blocks[3].attn, M.blocks[3].cross_attn
        cast = lambda a: mx.array(a).astype(M.dtype)
        chk = lambda key, out, sub=None: assert_close(
            R[key] if sub is None else R[key][sub], out, mode, tol,
            None if mode == "strict" else (BASE[key] if sub is None else BASE[key][sub]), name=f"{key}{sub or ''}")
        chk("attn50", ma(cast(c["x50"]), M._block_mask, 0))
        kv = {k: cast(v) for k, v in c["kv_self"].items()}
        chk("attn12", ma(cast(c["x12"]), M._block_mask, c["start"], kv_cache=kv, update_kv_cache=True))
        chk("kv", kv["k"], "k"); chk("kv", kv["v"], "v")
        kv50 = {k: cast(v) for k, v in c["kv_self"].items()}
        ma(cast(c["x50"]), M._block_mask, 0, kv_cache=kv50, update_kv_cache=True)
        chk("kv50", kv50["k"], "k")
        chk("cross50", mc(cast(c["x50"]), cast(c["x50"]), mask=M._alignment_mask))
        ckv = {k: cast(v) for k, v in c["kv_cross"].items()}
        chk("cross12", mc(cast(c["x12"]), cast(c["c12"]), mask=M._alignment_mask, kv_cache=ckv, update_kv_cache=True))
        chk("ckv", ckv["k"], "k")


@pytest.mark.parametrize("mode,tol", MODES)
def test_condition_and_adaLN(mlx_ft, module_case, mode, tol):
    c, R, BASE = module_case
    with device_for(mode):
        M = mlx_ft(mode)
        z = np.zeros_like(c["wa"])
        b = lambda k: None if mode == "strict" else BASE[k]
        assert_close(R["cond_null"], M.compute_condition(mx.array(c["wa"]), mx.array(z), mx.array(z)),
                     mode, tol, b("cond_null"), "cond_null")
        assert_close(R["cond_user"], M.compute_condition(mx.array(c["wa"]), mx.array(c["user"]), mx.array(c["motion"])),
                     mode, tol, b("cond_user"), "cond_user")
        ada = M.precompute_timestep_adaLN(c["ts"], 6, 10, context_len=2)
        for i in (0, 4, 8):
            for (j, k) in ((0, 0), (0, 5), (2, 1)):
                assert_close(R["ada"][i][j][k], ada[i][j][k], mode, tol,
                             None if mode == "strict" else BASE["ada"][i][j][k], f"ada[{i}][{j}][{k}]")


@pytest.fixture(scope="module")
def forward_case(torch_ft):
    T = torch_ft
    r = np.random.default_rng(3)
    B = 3
    c = dict(ts=np.linspace(1000, 0, 9).tolist(),
             x50=r.standard_normal((B, 50, 512)).astype(np.float32),
             x12=r.standard_normal((B, 12, 512)).astype(np.float32),
             wa50=audio_feats(r, B, 50), wa12=audio_feats(r, B, 12),
             u12=audio_feats(r, B, 12), m12=r.standard_normal((B, 12, 512)).astype(np.float32) * 0.3,
             wr=r.standard_normal((B, 512)).astype(np.float32),
             kv=[(_kv(r, B, 38), _kv(r, B, 2)) for _ in range(8)], start=98)

    def refs():
        R = {}
        with torch.no_grad():
            z50 = torch.zeros(B, 50, 512)
            c50 = T.compute_condition(t(c["wa50"]), z50, z50)
            c12 = T.compute_condition(t(c["wa12"]), t(c["u12"]), t(c["m12"]))
            ada50 = T.precompute_timestep_adaLN(c["ts"], B, 50, "cpu", 0)
            ada12 = T.precompute_timestep_adaLN(c["ts"], B, 10, "cpu", 2)
            R["primer"] = T.forward_with_precomputed(t(c["x50"]), c50, ada50[2], T.compute_wr_embedded(t(c["wr"]), 50),
                                                     0, None, False)
            R["kv"] = [tuple({k: t(v.copy()) for k, v in d.items()} for d in pair) for pair in c["kv"]]
            R["steady"] = T.forward_with_precomputed(t(c["x12"]), c12, ada12[5], T.compute_wr_embedded(t(c["wr"]), 12),
                                                     c["start"], R["kv"], True)
            t_tok = torch.cat([torch.zeros(B, 2), torch.full((B, 10), 250.0)], dim=1)
            R["full"] = T(t_tok, t(c["x12"]), t(c["wa12"]), t(c["u12"]), t(c["m12"]), t(c["wr"]), train=False,
                          start_pos=c["start"] + 10, kv_cache_list=R["kv"], update_kv_cache=False)
        return R
    R = refs()
    with torch_bf16():
        BASE = refs()
    return c, R, BASE


@pytest.mark.parametrize("mode,tol", FWD_MODES)
def test_forward_with_precomputed(mlx_ft, forward_case, mode, tol):
    c, R, BASE = forward_case
    b = lambda key, *idx: None if mode == "strict" else _get(BASE[key], idx)
    with device_for(mode):
        M = mlx_ft(mode)
        a = mx.array
        z50 = mx.zeros((3, 50, 512))
        out = M.forward_with_precomputed(a(c["x50"]), M.compute_condition(a(c["wa50"]), z50, z50),
                                         M.precompute_timestep_adaLN(c["ts"], 3, 50, 0)[2],
                                         M.compute_wr_embedded(a(c["wr"]), 50), 0)
        assert_close(R["primer"], out, mode, tol, b("primer"), "primer")
        kv = [tuple({k: a(v).astype(M.dtype) for k, v in d.items()} for d in pair) for pair in c["kv"]]
        out = M.forward_with_precomputed(a(c["x12"]), M.compute_condition(a(c["wa12"]), a(c["u12"]), a(c["m12"])),
                                         M.precompute_timestep_adaLN(c["ts"], 3, 10, 2)[5],
                                         M.compute_wr_embedded(a(c["wr"]), 12), c["start"], kv, True)
        assert_close(R["steady"], out, mode, tol, b("steady"), "steady")
        for i in (0, 7):
            assert_close(R["kv"][i][0]["k"], kv[i][0]["k"], mode, tol, b("kv", i, 0, "k"), f"self-kv[{i}]")
            assert_close(R["kv"][i][1]["v"], kv[i][1]["v"], mode, tol, b("kv", i, 1, "v"), f"cross-kv[{i}]")
        t_tok = mx.concatenate([mx.zeros((3, 2)), mx.full((3, 10), 250.0)], axis=1)
        out = M(t_tok, a(c["x12"]), a(c["wa12"]), a(c["u12"]), a(c["m12"]), a(c["wr"]),
                start_pos=c["start"] + 10, kv_cache_list=kv)
        assert_close(R["full"], out, mode, tol, b("full"), "full forward")


def _get(d, idx):
    for i in idx:
        d = d[i]
    return d
