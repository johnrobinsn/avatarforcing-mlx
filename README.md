# avatarforcing-mlx

**[AvatarForcing](https://github.com/TaekyungKi/AvatarForcing) (CVPR 2026) on Apple Silicon.**
Real-time, audio-driven talking-head video from a single portrait, on a MacBook — including
AvatarForcing's interactive mode, where the avatar reacts to its conversation partner's
voice and face.

> An independent port of the work of Taekyung Ki, Sangwon Jang, Jaehyeong Jo, Jaehong Yoon
> and Sung Ju Hwang — please cite [their paper](#citation). Not affiliated with or
> endorsed by the authors. **Non-commercial use only** (CC BY-NC 4.0, like the original).

## What it is

A hybrid of MLX and PyTorch, each used where it is fastest on Apple GPUs:

| Stage | Runs on | |
|---|---|---|
| Flow transformer + diffusion-forcing sampler | **MLX** (port) | the model that turns audio into motion |
| Motion decoder (latent → 512×512 frames) | PyTorch MPS, with [exact fast-path rewrites](src/avatarforcing_mlx/decoder_fast.py) + `torch.compile` | MLX convolutions are slower at this resolution |
| Motion encoder, wav2vec2 audio features | PyTorch MPS | the original modules |

On a **MacBook Pro M5 (base), 32 GB**, 25 fps at 512×512, nfe = 10:

| | |
|---|---|
| Per 10-frame block (400 ms of video) | **~210–235 ms** → 1.7–1.9× realtime |
| Flow transformer per block (MLX, fp32) | ~60 ms (2 CFG branches) / ~65 ms (3, interactive) |
| Motion decoder per block | ~150–165 ms (was 613 ms as shipped) |
| Frame accuracy vs the original fp32 PyTorch pipeline | per-frame SSIM 0.997–0.999 mean (PyTorch's own bf16: 0.96–0.98) |

The MLX code is verified against the authors' original PyTorch implementation (pinned
commit): the full sampling loop agrees to ~1e-5 in float32, in both talking-only and
interactive (user-conditioned) modes. Details: [docs/PERF.md](docs/PERF.md).

## Install

Apple Silicon Mac, Python 3.11–3.12, [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/johnrobinsn/avatarforcing-mlx && cd avatarforcing-mlx
uv sync                    # add --extra crop for automatic face cropping
uv run avatarforcing-mlx download
```

`download` fetches the original checkpoints from the AvatarForcing authors' Google Drive
and wav2vec2 from Hugging Face into `~/.cache/avatarforcing-mlx` (override with
`AVATARFORCING_MLX_HOME`), and converts the flow transformer to MLX locally (~1.4 GB in
total). Nothing is redistributed by this repo. If Google Drive rate-limits you, download
`flow_transformer.pth` and `motion_autoencoder.pth` manually from the
[AvatarForcing README](https://github.com/TaekyungKi/AvatarForcing#model-checkpoints) into
that directory and run `download` again.

## Use

```bash
# talking head: a face-centred portrait + speech
uv run avatarforcing-mlx generate --image face.jpg --audio speech.wav --out out.mp4

# interactive: also condition on the partner's voice and face (25 fps video)
uv run avatarforcing-mlx generate --image face.jpg --audio speech.wav \
    --user-audio partner.wav --user-video partner.mp4 --out out.mp4

# detect and crop faces first (as the original preprocessing does; needs --extra crop)
uv run avatarforcing-mlx generate --crop --image photo.jpg --audio speech.wav --out out.mp4
```

Options: `--nfe` (10), `--a-cfg` (2.0, audio guidance), `--u-cfg` (1.0, user guidance),
`--seed`, `--seconds`, `--flow-dtype float32|bfloat16`, `--decoder-dtype float16|float32`.
`uv run avatarforcing-mlx bench` times the flow transformer alone.

From Python:

```python
from avatarforcing_mlx import AvatarPipeline
from avatarforcing_mlx import media

pipe = AvatarPipeline()                                  # MLX fp32 flow, fp16 decoder
pipe.set_reference(media.load_image("face.jpg"))        # RGB uint8, 512x512
wa = pipe.audio_features(media.load_audio("speech.wav"))
frames = [media.to_uint8(pix) for _, pix, _ in pipe.stream(wa)]   # ~10 frames per block
```

`stream()` yields frames block by block as they are generated (a 2 s primer, then 0.4 s
blocks), so it can drive a live display. For a full real-time voice chat built on this,
see [AvatarVoice](https://github.com/johnrobinsn/AvatarVoice).

## Notes

- **Precision.** The flow transformer runs in float32 by default. bfloat16 is ~25 ms per
  block faster, but the diffusion sampler amplifies its rounding — exactly as it does for
  PyTorch's own bf16 — and frames drift further from the reference (SSIM ~0.976). The
  decoder runs in fp16 (SSIM cost < 0.001).
- **Fidelity.** The sampler implements AvatarForcing's 3-branch classifier-free guidance
  (unconditional / avatar audio / user). Without user inputs the user branch provably has
  no effect, so it is skipped (2 branches, same result).
- **Long sessions.** Rotary position encoding uses `mx.fast.rope` with an offset, so there
  is no 1024-position limit (the PyTorch code precomputes a 1024-entry table).
- MLX streams are thread-local: create and use an `AvatarPipeline` on the same thread.
- `transformers` is pinned to 4.30.2, which AvatarForcing's wav2vec2 wrapper targets.

## Tests

The tests compare the MLX port with the original PyTorch code at the pinned commit:

```bash
scripts/fetch_upstream.sh          # clones TaekyungKi/AvatarForcing into third_party/
uv sync --group dev
uv run pytest                      # ~30 s; needs the downloaded model files
```

They cover the RoPE and attention masks, every module (strict fp32 on the MLX CPU
stream, and bf16 against PyTorch's own bf16 error), the full forward pass with the KV
cache, the complete sampling loop against AvatarForcing's `solve_cfg` / `solve_closed` /
`update_kv_cache` in both modes, the decoder rewrites (float64, exact), and that the
vendored PyTorch files match upstream byte for byte apart from import paths.

## License and attribution

CC BY-NC 4.0 ([LICENSE](LICENSE)) — non-commercial use only. This is an adaptation of
AvatarForcing; [NOTICE](NOTICE) lists what was changed and credits the work it builds on:
AvatarForcing; FLOAT (Ki, Min, Chae — the motion autoencoder derives from it; its
checkpoints are CC BY-NC-ND 4.0, research use); LIA (Wang, Yang, Bremond, Dantcheva);
and wav2vec 2.0 (Apache 2.0).

## Citation

Please cite AvatarForcing:

```bibtex
@InProceedings{Ki_2026_CVPR,
    author    = {Ki, Taekyung and Jang, Sangwon and Jo, Jaehyeong and Yoon, Jaehong and Hwang, Sung Ju},
    title     = {Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {18074-18084}
}
```
