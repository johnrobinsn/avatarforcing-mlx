"""Frame-level comparison with the original PyTorch pipeline.

Renders the same portrait + speech (same noise) twice:
  reference — AvatarForcing's own sampling code (third_party/AvatarForcing, CPU fp32)
              + the original motion decoder (fp32, MPS, no fast kernels)
  mlx       — AvatarPipeline (MLX flow transformer + fast fp16 decoder)
and prints per-frame SSIM.  Optionally also a PyTorch-bf16 reference, for scale.

    uv run python scripts/compare_reference.py --image face.jpg --audio speech.wav --seconds 6
"""
from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "AvatarForcing"))

from avatarforcing_mlx import decoder_fast, media, weights
from avatarforcing_mlx.pipeline import AvatarPipeline, image_to_tensor, load_generator, load_opt

CLIP, BLK = 50, 10


def ssim(a: np.ndarray, b: np.ndarray, device="mps") -> np.ndarray:
    """Per-frame SSIM (11x11 Gaussian, sigma 1.5) on uint8 [T, H, W, 3], channel-averaged."""
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = ((g[:, None] * g[None, :]) / g.sum() ** 2).view(1, 1, 11, 11).repeat(3, 1, 1, 1).to(device)
    out = []
    for i in range(0, len(a), 16):
        x = torch.from_numpy(a[i:i + 16]).permute(0, 3, 1, 2).float().to(device)
        y = torch.from_numpy(b[i:i + 16]).permute(0, 3, 1, 2).float().to(device)
        f = lambda z: F.conv2d(z, w, groups=3)
        mx_, my = f(x), f(y)
        sxx, syy, sxy = f(x * x) - mx_ ** 2, f(y * y) - my ** 2, f(x * y) - mx_ * my
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        s = ((2 * mx_ * my + c1) * (2 * sxy + c2)) / ((mx_ ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        out.append(s.mean(dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


def reference_frames(image, wa, noises, T, bf16=False) -> np.ndarray:
    import models.avatarforcing.flow_transformer as tft
    from models import FlowMatchScheduler
    from models.avatarforcing.AvatarForcing import AvatarForcing
    p = weights.paths()
    opt = load_opt(p["wav2vec2"]); opt.rank = "cpu"
    ft = tft.FlowTransformer(opt)
    sd = torch.load(p["flow_transformer"], map_location="cpu", weights_only=True)
    with torch.no_grad():
        for k, prm in ft.named_parameters():
            prm.copy_(sd["flow_transformer." + k])
    ft.eval()
    for blk in ft.blocks:      # extend RoPE past 1024 positions for long clips
        blk.attn.freqs_cis = tft.precompute_freqs_cis(blk.attn.head_dim, 65536)
    af = types.SimpleNamespace(opt=opt, rank="cpu", flow_transformer=ft, num_frames_for_clip=CLIP,
                               num_frames_per_block=BLK)
    af.scheduler = FlowMatchScheduler(num_steps=1000, shift=1, sigma_min=0.0, extra_one_step=False)
    af.scheduler.set_timesteps(1000, training=True)
    af.denoising_step_list = torch.tensor(np.linspace(1000, 0, 9).tolist())
    for n in ("solve_cfg", "solve_closed", "prepare_cfg_condition", "update_kv_cache", "initialize_kv_cache"):
        setattr(af, n, types.MethodType(getattr(AvatarForcing, n), af))

    decoder_fast.unpatch()
    g = load_generator(p["motion_autoencoder"], "mps")
    with torch.inference_mode():
        s_r, _, feats = g.enc(image_to_tensor(image, "mps"), input_target=None)
        r_s = g.dec.direction(g.enc.fc(s_r)).cpu()
    n_blocks = len(noises) - 1
    total = CLIP + n_blocks * BLK + 2
    wa = np.concatenate([wa, np.repeat(wa[:, -1:], total - wa.shape[1], axis=1)], axis=1)
    z = lambda L: torch.zeros(1, L, 512)
    lat = []
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        af.initialize_kv_cache(1, torch.float32, "cpu")

        def run(x_t, lo, L, seq, ctx, sp, kv):
            pc = af.prepare_cfg_condition(torch.from_numpy(wa[:, lo:lo + L]), z(L), z(L), r_s, seq_len=seq, context_len=ctx)
            for i, ts in enumerate(af.denoising_step_list):
                x_t = af.solve_cfg(B=1, index=i, current_timestep=ts, x_t=x_t, precomputed_c=pc[0], precomputed_wr=pc[1],
                                   precomputed_adaLN=pc[2], start_pos=sp, context_len=ctx, use_kv_cache=kv,
                                   a_cfg_scale=2.0, u_cfg_scale=1.0)
            af.update_kv_cache(final_latents=x_t, precomputed_c=pc[0], precomputed_wr=pc[1],
                               precomputed_adaLN=pc[2], start_pos=sp)
            return x_t
        x_t = run(torch.from_numpy(noises[0]), 0, CLIP, CLIP, 0, 0, False)
        lat.append(x_t.float().numpy())
        for b in range(n_blocks):
            ti = CLIP + b * BLK
            x_t = run(torch.cat([x_t[:, -2:], torch.from_numpy(noises[b + 1])], dim=1), ti - 2, BLK + 2, BLK, 2, ti - 2, True)
            lat.append(x_t[:, -BLK:].float().numpy())
    lat = np.concatenate(lat, axis=1)[:, :T]
    fe = [f.repeat_interleave(BLK, 0) for f in feats]
    frames = []
    with torch.inference_mode():
        for b in range(0, T, BLK):
            zb = torch.from_numpy(lat[:, b:b + BLK]).to("mps")
            k = zb.shape[1]
            if k < BLK:
                zb = torch.cat([zb, zb[:, -1:].expand(1, BLK - k, -1)], dim=1)
            img, _ = g.dec((s_r.unsqueeze(1) + zb).reshape(BLK, -1), alpha=None, feats=fe)
            frames.append(media.to_uint8(img[:k]))
    return np.concatenate(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--torch-bf16", action="store_true", help="also report PyTorch-bf16 vs reference")
    args = ap.parse_args()

    image = media.load_image(args.image)
    pcm = media.load_audio(args.audio)[: int(args.seconds * media.SAMPLE_RATE)]
    T = int(np.ceil(len(pcm) * media.FPS / media.SAMPLE_RATE))
    rng = np.random.default_rng(args.seed)
    noises = [rng.standard_normal((1, CLIP, 512)).astype(np.float32)] + \
             [rng.standard_normal((1, BLK, 512)).astype(np.float32) for _ in range(len(range(CLIP, T, BLK)))]

    pipe = AvatarPipeline()
    pipe.set_reference(image)
    wa = pipe.audio_features(pcm, T)
    wa_np = np.array(wa)
    out = np.concatenate([media.to_uint8(pix) for _, pix, _ in pipe.stream(wa, noises=noises)])[:T]

    t0 = time.perf_counter()
    ref = reference_frames(image, wa_np, noises, T)
    print(f"reference rendered in {time.perf_counter() - t0:.1f}s")
    s = ssim(out, ref)
    print(f"MLX vs original fp32:   SSIM mean {s.mean():.4f}  min {s.min():.4f}  frames >= 0.995: {(s >= 0.995).mean():.0%}")
    if args.torch_bf16:
        s = ssim(reference_frames(image, wa_np, noises, T, bf16=True), ref)
        print(f"PyTorch bf16 vs fp32:   SSIM mean {s.mean():.4f}  min {s.min():.4f}  frames >= 0.995: {(s >= 0.995).mean():.0%}")


if __name__ == "__main__":
    main()
