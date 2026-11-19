"""Full sampling loop (primer + streaming blocks) vs AvatarForcing's own
solve_cfg / solve_closed / prepare_cfg_condition / update_kv_cache."""
import numpy as np
import pytest
import torch
import mlx.core as mx

from avatarforcing_mlx.sampler import FlowSamplerMLX

from conftest import audio_feats, device_for, rel_rms, t, to_np, torch_bf16

N_BLOCKS, CLIP, BLK = 4, 50, 10


def torch_stream_latents(af, wa, user_wa, user_motion, wr, noises, a_cfg, u_cfg):
    """Latent path of AvatarForcing.sample (3-branch CFG, KV-cached blocks)."""
    af.initialize_kv_cache(batch_size=1, dtype=torch.float32, device="cpu")
    out, steps = [], af.denoising_step_list
    win = lambda a, lo, L: t(a[:, lo:lo + L])

    def run(x_t, lo, L, seq_len, ctx, start_pos, use_kv):
        pc = af.prepare_cfg_condition(win(wa, lo, L), win(user_wa, lo, L), win(user_motion, lo, L), t(wr),
                                      seq_len=seq_len, context_len=ctx)
        for i, ts in enumerate(steps):
            x_t = af.solve_cfg(B=1, index=i, current_timestep=ts, x_t=x_t, precomputed_c=pc[0],
                               precomputed_wr=pc[1], precomputed_adaLN=pc[2], start_pos=start_pos,
                               context_len=ctx, use_kv_cache=use_kv, a_cfg_scale=a_cfg, u_cfg_scale=u_cfg)
        af.update_kv_cache(final_latents=x_t, precomputed_c=pc[0], precomputed_wr=pc[1],
                           precomputed_adaLN=pc[2], start_pos=start_pos)
        return x_t

    with torch.inference_mode():
        x_t = run(t(noises[0]), 0, CLIP, CLIP, 0, 0, False)
        out.append(x_t.numpy())
        for b in range(N_BLOCKS):
            ti = CLIP + b * BLK
            x_t = torch.cat([x_t[:, -2:], t(noises[b + 1])], dim=1)
            x_t = run(x_t, ti - 2, BLK + 2, BLK, 2, ti - 2, True)
            out.append(x_t.numpy())
    return out


def mlx_stream_latents(sampler, wa, user_wa, user_motion, wr, noises):
    a = lambda x, lo, L: None if x is None else mx.array(x[:, lo:lo + L])
    sampler.reset(1, with_user=user_wa is not None)
    x_t = sampler.prime(a(wa, 0, CLIP), mx.array(wr), mx.array(noises[0]), a(user_wa, 0, CLIP), a(user_motion, 0, CLIP))
    mx.eval(x_t)
    out = [to_np(x_t)]
    for b in range(N_BLOCKS):
        ti = CLIP + b * BLK
        x_t = sampler.step(a(wa, ti - 2, BLK + 2), mx.array(wr), x_t, mx.array(noises[b + 1]), ti - 2,
                           a(user_wa, ti - 2, BLK + 2), a(user_motion, ti - 2, BLK + 2))
        mx.eval(x_t)
        out.append(to_np(x_t))
    return out


@pytest.fixture(scope="module")
def inputs():
    r = np.random.default_rng(4)
    T = CLIP + N_BLOCKS * BLK
    return dict(wa=audio_feats(r, 1, T), user_wa=audio_feats(r, 1, T),
                user_motion=(r.standard_normal((1, T, 512)) * 0.3).astype(np.float32),
                wr=r.standard_normal((1, 512)).astype(np.float32),
                noises=[r.standard_normal((1, CLIP, 512)).astype(np.float32)] +
                       [r.standard_normal((1, BLK, 512)).astype(np.float32) for _ in range(N_BLOCKS)])


def _assert_strict(ref, out, tol=1e-3):
    for b, (r, o) in enumerate(zip(ref, out)):
        d, scale = float(np.max(np.abs(r - o))), float(np.max(np.abs(r)))
        assert d <= tol * max(1.0, scale), f"block {b}: max_abs {d:.2e}"


@pytest.mark.slow
def test_no_user_two_branch_matches_three_branch_reference(upstream_af, mlx_ft, inputs):
    """No user inputs: MLX runs 2 CFG branches; the reference runs its usual 3 with null
    user conditions and u_cfg=1.  The user branch then equals the uncond branch, so the
    results must agree."""
    z = np.zeros_like(inputs["wa"])
    ref = torch_stream_latents(upstream_af(), inputs["wa"], z, z, inputs["wr"], inputs["noises"], 2.0, 1.0)
    with device_for("strict"):
        for skip in (False, True):
            s = FlowSamplerMLX(mlx_ft("strict"), nfe=10, a_cfg_scale=2.0, u_cfg_scale=1.0, skip_noop_steps=skip)
            out = mlx_stream_latents(s, inputs["wa"], None, None, inputs["wr"], inputs["noises"])
            assert s.branches == 2
            _assert_strict(ref, out)


@pytest.mark.slow
def test_user_conditioning_three_branch(upstream_af, mlx_ft, inputs):
    """Interactive mode: user audio + user motion, 3 CFG branches, u_cfg=1."""
    ref = torch_stream_latents(upstream_af(), inputs["wa"], inputs["user_wa"], inputs["user_motion"],
                               inputs["wr"], inputs["noises"], 2.0, 1.0)
    with device_for("strict"):
        s = FlowSamplerMLX(mlx_ft("strict"), nfe=10, a_cfg_scale=2.0, u_cfg_scale=1.0)
        out = mlx_stream_latents(s, inputs["wa"], inputs["user_wa"], inputs["user_motion"], inputs["wr"], inputs["noises"])
        assert s.branches == 3
    _assert_strict(ref, out)
    # and the user branch matters: dropping it changes the result
    z = np.zeros_like(inputs["wa"])
    ref_null = torch_stream_latents(upstream_af(), inputs["wa"], z, z, inputs["wr"], inputs["noises"], 2.0, 1.0)
    assert rel_rms(ref[-1], ref_null[-1]) > 1e-2


@pytest.mark.slow
def test_bf16_drift_matches_torch_bf16(upstream_af, mlx_ft, inputs):
    """bf16 sampling drifts from fp32 (diffusion + CFG amplify rounding); MLX-bf16 must
    stay within 1.5x (+0.01) of PyTorch-bf16's own drift."""
    args = (inputs["wa"], inputs["user_wa"], inputs["user_motion"], inputs["wr"], inputs["noises"])
    ref = torch_stream_latents(upstream_af(), *args, 2.0, 1.0)
    with torch_bf16():
        base = torch_stream_latents(upstream_af(), *args, 2.0, 1.0)
    with device_for("bf16"):
        out = mlx_stream_latents(FlowSamplerMLX(mlx_ft("bf16"), nfe=10, a_cfg_scale=2.0, u_cfg_scale=1.0), *args)
    for b in range(len(ref)):
        assert rel_rms(ref[b], out[b]) <= 1.5 * rel_rms(ref[b], base[b]) + 0.01, f"block {b}"
