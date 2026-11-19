"""avatarforcing-mlx command line.

    avatarforcing-mlx download                       # original checkpoints + MLX conversion
    avatarforcing-mlx generate --image face.jpg --audio speech.wav --out out.mp4
    avatarforcing-mlx generate ... --user-audio partner.wav --user-video partner.mp4
    avatarforcing-mlx bench                          # flow-transformer per-block timing
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np


def _generate(args):
    import mlx.core as mx
    from . import media
    from .pipeline import AvatarPipeline, StreamStats

    if args.crop:
        from PIL import Image
        from .preprocess import crop_face
        image = crop_face(np.array(Image.open(args.image).convert("RGB")))
    else:
        image = media.load_image(args.image)

    pcm = media.load_audio(args.audio)
    if args.seconds:
        pcm = pcm[: int(args.seconds * media.SAMPLE_RATE)]
    T = int(np.ceil(len(pcm) * media.FPS / media.SAMPLE_RATE))

    t0 = time.perf_counter()
    pipe = AvatarPipeline(args.weights, flow_dtype=args.flow_dtype, decoder_dtype=args.decoder_dtype,
                          compile_decoder=not args.no_compile, nfe=args.nfe,
                          a_cfg_scale=args.a_cfg, u_cfg_scale=args.u_cfg)
    pipe.set_reference(image)
    print(f"pipeline ready in {time.perf_counter() - t0:.1f}s")

    wa = pipe.audio_features(pcm, T)
    user_wa = user_motion = None
    if args.user_audio:
        upcm = media.load_audio(args.user_audio)[: len(pcm)]
        user_wa = pipe.audio_features(upcm, T)
    if args.user_video:
        frames = media.load_video_frames(args.user_video)[:T]
        if args.crop:
            from .preprocess import crop_video
            frames = crop_video(frames)
        elif frames.shape[1:3] != (512, 512):
            from PIL import Image
            frames = np.stack([np.array(Image.fromarray(f).resize((512, 512), Image.BICUBIC)) for f in frames])
        user_motion = pipe.user_motion(frames)
    mx.eval(wa)

    stats, out = StreamStats(), []
    t_start = time.perf_counter()
    for _, pix, _ in pipe.stream(wa, user_wa, user_motion, seed=args.seed, stats=stats):
        out.append(media.to_uint8(pix))
    wall = time.perf_counter() - t_start
    frames_out = np.concatenate(out)[:T]
    media.write_video(args.out, frames_out, pcm)

    if stats.timings:
        tot = sorted(t.total_ms for t in stats.timings)
        print(f"{len(stats.timings)} steady blocks: flow {statistics.median(t.flow_ms for t in stats.timings):.0f} ms + "
              f"decode {statistics.median(t.decode_ms for t in stats.timings):.0f} ms; total median {tot[len(tot) // 2]:.0f} "
              f"max {tot[-1]:.0f} EWMA {stats.ewma_ms:.0f} ms/block ({400 / stats.ewma_ms:.2f}x realtime)")
    print(f"{T / media.FPS:.1f}s of video in {wall:.1f}s -> {args.out}")


def _bench(args):
    import mlx.core as mx
    from . import weights
    from .flow_transformer import FlowTransformerMLX, load_weights
    from .pipeline import load_opt
    from .sampler import FlowSamplerMLX

    p = weights.paths(args.weights)
    model = FlowTransformerMLX(load_opt())
    load_weights(model, str(p["mlx"]), dtype=getattr(mx, args.dtype))
    s = FlowSamplerMLX(model, nfe=args.nfe, u_cfg_scale=1.0)
    rng = np.random.default_rng(0)
    n = args.iters + 3
    arr = lambda *shape: mx.array(rng.standard_normal(shape).astype(np.float32))
    wa, wr = arr(1, 50 + n * 10, 512), arr(1, 512)
    user = (arr(1, 50 + n * 10, 512), arr(1, 50 + n * 10, 512)) if args.user else (None, None)
    s.reset(1, with_user=args.user)
    t0 = time.perf_counter()
    x = s.prime(wa[:, :50], wr, arr(1, 50, 512), *(u[:, :50] if u is not None else None for u in user))
    mx.eval(x, s.kv_cache)
    print(f"primer: {(time.perf_counter() - t0) * 1000:.0f} ms (first call)")
    times = []
    for b in range(n):
        t = 50 + 10 * b
        t0 = time.perf_counter()
        x = s.step(wa[:, t - 2:t + 10], wr, x, arr(1, 10, 512), t - 2,
                   *(u[:, t - 2:t + 10] if u is not None else None for u in user))
        mx.eval(x, s.kv_cache)
        if b >= 3:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"flow transformer, {args.dtype}, nfe={args.nfe}, {s.branches} CFG branches, "
          f"{len(s.steps) + 1} forward passes/block: median {times[len(times) // 2]:.1f} ms, "
          f"min {times[0]:.1f}, max {times[-1]:.1f} (n={len(times)})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="avatarforcing-mlx", description="AvatarForcing on Apple Silicon (MLX)")
    ap.add_argument("--weights", default=None, help="model directory (default $AVATARFORCING_MLX_HOME "
                                                    "or ~/.cache/avatarforcing-mlx)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch the original checkpoints and convert the flow transformer")
    d.add_argument("--force", action="store_true")

    c = sub.add_parser("convert", help="convert a flow_transformer.pth to MLX safetensors")
    c.add_argument("src")
    c.add_argument("dst")

    g = sub.add_parser("generate", help="portrait + speech -> talking-head video")
    g.add_argument("--image", required=True, help="avatar portrait (face-centred; or pass --crop)")
    g.add_argument("--audio", required=True, help="avatar speech (any format PyAV reads)")
    g.add_argument("--user-audio", help="conversation partner's audio (interactive mode)")
    g.add_argument("--user-video", help="conversation partner's video, 25 fps (interactive mode)")
    g.add_argument("--crop", action="store_true", help="face-detect and crop image / user video "
                                                       "(needs the 'crop' extra)")
    g.add_argument("--out", default="out.mp4")
    g.add_argument("--seconds", type=float, help="truncate the audio")
    g.add_argument("--seed", type=int, default=25)
    g.add_argument("--nfe", type=int, default=10)
    g.add_argument("--a-cfg", type=float, default=2.0, help="avatar-audio guidance scale")
    g.add_argument("--u-cfg", type=float, default=1.0, help="user guidance scale (used with --user-*)")
    g.add_argument("--flow-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    g.add_argument("--decoder-dtype", default="float16", choices=["float16", "float32"])
    g.add_argument("--no-compile", action="store_true", help="skip torch.compile of the decoder")

    b = sub.add_parser("bench", help="flow-transformer per-block timing")
    b.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    b.add_argument("--nfe", type=int, default=10)
    b.add_argument("--iters", type=int, default=20)
    b.add_argument("--user", action="store_true", help="3-branch CFG (with user conditioning)")

    args = ap.parse_args(argv)
    if args.cmd == "download":
        from . import weights
        p = weights.download(args.weights, force=args.force)
        print(f"model files ready in {p['root']}")
    elif args.cmd == "convert":
        from . import weights
        weights.convert(args.src, args.dst)
    elif args.cmd == "generate":
        _generate(args)
    elif args.cmd == "bench":
        _bench(args)


if __name__ == "__main__":
    sys.exit(main())
