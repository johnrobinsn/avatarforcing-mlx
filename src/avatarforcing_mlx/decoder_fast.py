# Performance rewrites of functions in AvatarForcing's models/avatarforcing/styledecoder.py
# (https://github.com/TaekyungKi/AvatarForcing, commit 507e3046ad53), which builds on LIA and
# FLOAT. Copyright the original authors; rewrites by John Robinson. CC BY-NC 4.0 (see NOTICE).

"""Faster motion-decoder kernels for torch MPS — exact algebraic rewrites.

`patch()` replaces hot spots in the vendored torch_models/avatarforcing/styledecoder.py
(which itself is left unmodified; this module swaps functions at runtime, opt-in):

1. ModulatedConv2d — the original folds the per-sample style into the conv
   weight and runs a *grouped* conv (groups=batch), plus a grouped
   conv_transpose2d for upsampling.  Grouped per-sample convs are slow on MPS.
   Rewrite (StyleGAN2 "fused_modconv=False"):
       out = conv(x * style, W) * demod
   which is the same math: style scales input channels, demod scales output
   channels, and both commute with the (linear) convolution.  One shared-weight
   conv over the whole batch.

2. upfirdn2d — the original reshapes to (B*C, 1, H, W) and runs a
   single-channel conv.  Rewrite as one depthwise conv (groups=C), using a
   strided depthwise conv_transpose2d for the zero-insertion upsample.

3. ToFlow — the original rebuilds its sampling grid with numpy and copies it
   from the CPU on every call, always as float32 (which also breaks pure-fp16
   execution in grid_sample).  The grid is now built once per (size, dtype,
   device) on the device.

Running the decoder in pure fp16 (`decoder.half()`, fp16 feats) rather than
under autocast also matters: under autocast, FusedLeakyReLU's fp32 bias promotes
every activation back to fp32, adding cast passes over 512x512 tensors.

`unpatch()` restores the originals.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .torch_models.avatarforcing import styledecoder as sd

_ORIG = {}


def _modconv_forward(self, input, style):
    batch, in_channel, height, width = input.shape
    style = self.modulation(style)                                   # (B, Cin)
    weight = self.scale * self.weight[0]                             # (Cout, Cin, k, k)
    x = input * style.view(batch, in_channel, 1, 1)

    if self.upsample:
        # conv_transpose2d weight layout is (Cin, Cout, k, k)
        out = F.conv_transpose2d(x, weight.transpose(0, 1), padding=0, stride=2)
    elif self.downsample:
        out = F.conv2d(self.blur(x), weight, padding=0, stride=2)
    else:
        out = F.conv2d(x, weight, padding=self.padding)

    if self.demodulate:
        # demod[b, o] = rsqrt(sum_{i,kh,kw} (W[o,i,kh,kw] * style[b,i])^2)
        w2 = weight.pow(2).sum([2, 3])                               # (Cout, Cin)
        demod = torch.rsqrt(style.pow(2) @ w2.t() + 1e-8)            # (B, Cout)
        out = out * demod.view(batch, self.out_channel, 1, 1)

    if self.upsample:
        out = self.blur(out)
    return out


def _upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    B, C, H, W = input.shape
    kh, kw = kernel.shape
    p0, p1 = pad
    w = torch.flip(kernel, [0, 1]).to(input.dtype).view(1, 1, kh, kw).expand(C, 1, kh, kw)
    if up > 1:
        # Zero-insertion upsample + pad + correlation == strided transposed conv.
        # Transposed conv with kernel K correlates with flip(K); pass the
        # unflipped kernel so the effective filter is the flipped one above.
        wt = kernel.to(input.dtype).view(1, 1, kh, kw).expand(C, 1, kh, kw)
        # Output of conv_transpose2d (no padding): (H-1)*up + kh.  The native path
        # yields H*up + p0 + p1 - kh + 1 (with up-1 trailing zeros); crop/pad to match.
        out = F.conv_transpose2d(input, wt, stride=up, groups=C)
        lo = kh - 1 - p0
        hi_len = H * up + p0 + p1 - kh + 1
        if lo < 0:
            out = F.pad(out, [-lo, 0, -lo, 0])
            lo = 0
        need = lo + hi_len
        if need > out.shape[-1]:
            out = F.pad(out, [0, need - out.shape[-1], 0, need - out.shape[-2]])
        out = out[:, :, lo:lo + hi_len, lo:lo + hi_len]
    else:
        out = F.pad(input, [max(p0, 0), max(p1, 0), max(p0, 0), max(p1, 0)])
        if p0 < 0 or p1 < 0:
            out = out[:, :, max(-p0, 0):out.shape[2] - max(-p1, 0), max(-p0, 0):out.shape[3] - max(-p1, 0)]
        out = F.conv2d(out, w, groups=C)
    return out[:, :, ::down, ::down]


_GRIDS: dict = {}


def _base_grid(n, dtype, device):
    """np.stack(np.meshgrid(linspace(-1,1,n), linspace(-1,1,n)), 2) -> (1, n, n, 2)."""
    key = (n, dtype, str(device))
    if key not in _GRIDS:
        xs = torch.linspace(-1, 1, n, dtype=torch.float64)
        gx, gy = torch.meshgrid(xs, xs, indexing="xy")
        # The original always builds this grid in float32 (`torch.tensor(xs).float()`),
        # whatever the model dtype; keep that rounding so results match exactly.
        grid = torch.stack([gx, gy], dim=2).unsqueeze(0).to(torch.float32)
        _GRIDS[key] = grid.to(device=device, dtype=dtype)
    return _GRIDS[key]


def _toflow_forward(self, input, style, feat, skip=None):
    out = self.conv(input, style)
    out = out + self.bias
    if skip is not None:
        out = out + self.upsample(skip)
    sampler = torch.tanh(out[:, 0:2, :, :])
    mask = torch.sigmoid(out[:, 2:3, :, :])
    flow = sampler.permute(0, 2, 3, 1) + _base_grid(input.size(2), sampler.dtype, input.device)
    feat_warp = F.grid_sample(feat.to(flow.dtype), flow, align_corners=False) * mask
    return feat_warp, feat_warp + input * (1.0 - mask), out, flow


def patch():
    if _ORIG:
        return
    _ORIG["modconv"] = sd.ModulatedConv2d.forward
    _ORIG["upfirdn2d"] = sd.upfirdn2d
    _ORIG["toflow"] = sd.ToFlow.forward
    sd.ModulatedConv2d.forward = _modconv_forward
    sd.upfirdn2d = _upfirdn2d
    sd.ToFlow.forward = _toflow_forward


def unpatch():
    if not _ORIG:
        return
    sd.ModulatedConv2d.forward = _ORIG.pop("modconv")
    sd.upfirdn2d = _ORIG.pop("upfirdn2d")
    sd.ToFlow.forward = _ORIG.pop("toflow")
