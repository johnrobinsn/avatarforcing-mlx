# Performance and accuracy

Machine: MacBook Pro, **Apple M5 (base), 32 GB**, macOS 26.6; MLX 0.32.4, PyTorch 2.15.0.
Settings: 25 fps, 512×512, nfe = 10 (9 denoising steps, the last a no-op), a_cfg = 2.0.
Timings vary ~10% with background load (the 210 ms runs were on an idle machine).

## End to end

`avatarforcing-mlx generate`, 15.8 s of speech, steady-state 10-frame blocks (400 ms of video):

| | per block |
|---|---|
| Flow transformer + sampler (MLX fp32) | ~58–65 ms |
| Motion decoder (PyTorch MPS fp16, compiled) | ~150–165 ms |
| **Total** | **~210–235 ms (1.7–1.9× realtime)**, max within ~15 ms of the median |
| Interactive mode (3 CFG branches) | +~3 ms |
| 50-frame primer (KV re-anchor), flow only | ~68 ms |
| Unified memory | ~0.8 GB MLX + ~4.7 GB MPS |

For reference, the PyTorch/CUDA path on 2× RTX Pro 6000 with CUDA graphs: flow
transformer ~135 ms per block; the full live pipeline ~210 ms.

## Flow transformer (MLX) — `avatarforcing-mlx bench`

| dtype | per block |
|---|---|
| float32 | 57–63 ms |
| bfloat16 | 37.5 ms |
| float16 | 37.4 ms |

The flow transformer is memory-bound (2–3 × 12 tokens per pass), which is why a third
CFG branch is nearly free. The final denoising step (t = 0 → 0) leaves every token
unchanged in `solve_closed`, so its forward pass is skipped (~10% saving, identical
output).

## Motion decoder (PyTorch MPS)

| Variant | 10 frames @ 512 |
|---|---|
| original, fp32 | 613 ms |
| original, fp16 autocast | 510 ms |
| fast kernels, fp16 autocast | 458 ms |
| fast kernels, pure fp16 | 333 ms |
| fast kernels, pure fp16, `torch.compile` | **150 ms** |

The decoder's convolutions are only ~60–70 ms of work at the ~12 TFLOP/s MPS sustains on
these shapes. The rest was: per-sample grouped modulated convolutions; `upfirdn2d` as a
single-channel convolution over B·C maps; autocast promoting activations back to fp32 at
an fp32 bias; a sampling grid rebuilt with numpy and copied from the CPU on every call;
and un-fused elementwise passes over 170 MB tensors. `decoder_fast.py` rewrites the first
three exactly (float64 agreement ~1e-14 per operation); pure fp16 removes the casts;
`torch.compile` (Inductor's MPS backend) fuses the elementwise chains.

MLX convolutions were evaluated for a decoder port and were *slower* than MPS at high
resolution (64→32 channels at 512²: MLX 57 ms, MPS 7.6 ms), so the decoder stays on PyTorch.

## Accuracy

Strict (MLX float32 on the CPU stream vs the original PyTorch float32): modules and full
forward passes within ~1e-5 relative; the full sampling loop (primer + streaming blocks,
with KV cache) within 1e-3 absolute on the latents, in both 2-branch and 3-branch modes.
The strict tests use MLX's CPU stream because fp32 matrix multiplies on the M5 GPU carry
~1e-3 relative error (vs ~1e-6 on the CPU).

Frames vs the fp32 PyTorch reference (original sampling + original fp32 decoder),
same noise, per-frame SSIM — reproduce with `scripts/compare_reference.py`. On
AvatarForcing's own sample (`data/rumi.jpg` + `data/avatar.wav`, 6 s) the default
settings give mean **0.9987**, min 0.9982, 100% of frames ≥ 0.995 (PyTorch bf16: mean
0.963). On a second 6 s clip:

| Flow / decoder | per block | SSIM mean | SSIM min | frames ≥ 0.995 |
|---|---|---|---|---|
| MLX bf16 / fp16 | 189 ms | 0.976 | 0.916 | 23% |
| **MLX fp32 / fp16 (default)** | **214 ms** | **0.9975** | **0.9943** | **89%** |
| MLX fp32 / fp32 | 372 ms | 0.9982 | 0.9951 | 100% |
| PyTorch bf16 autocast | — | 0.981 | 0.952 | 35% |

bf16 rounding is amplified by the diffusion sampler with CFG; PyTorch's own bf16 drifts
by the same amount (latent relative RMS up to ~27% mid-stream for both). Hence the fp32
default.

## Notes from the port

- The checkpoint's `alignment_mask` / `block_bidir_causal_lookahead_mask` buffers are the
  *inverse* of the masks the code generates. The original inference loads parameters
  only, so the generated masks are what the model actually uses; loading the checkpoint
  with `load_state_dict(strict=True)` would silently invert attention. The converter
  drops them.
- The checkpoints are float32.
- `mx.fast.scaled_dot_product_attention` accepts boolean masks, and
  `mx.fast.rope(traditional=True, offset=start_pos)` matches the complex-polar RoPE.
- transformers' warning that wav2vec2 `pos_conv_embed.conv.weight_g/weight_v` were "not
  used" is spurious: torch's weight-norm parametrization remaps them on load.
