# MLX port of AvatarForcing (https://github.com/TaekyungKi/AvatarForcing), an adaptation of
# the sampling loop in models/avatarforcing/AvatarForcing.py at commit 507e3046ad53.
# Copyright the AvatarForcing authors; MLX port by John Robinson. Licensed CC BY-NC 4.0
# (see LICENSE and NOTICE).

"""Diffusion-forcing sample loop for the MLX flow transformer.

Mirrors the latent path of `AvatarForcing.sample` / `inference_stream`: a 50-frame
primer, then 10-frame blocks with 2 frames of overlap, using `prepare_cfg_condition`,
`solve_cfg`, `solve_closed` and `update_kv_cache`.  Pixel decoding is not included.

Classifier-free guidance, as in AvatarForcing (3 branches):

    branches = [uncond, avatar audio, user audio + user motion]
    v = v_uncond + a_cfg * (v_aud - v_uncond) + u_cfg * (v_user - v_uncond)

Without user inputs the user branch is conditioned on zeros — the same as the
uncond branch — so v_user == v_uncond and its term vanishes; with u_cfg == 0 it is
multiplied away.  Either way, since CFG branches never interact (each batch element
attends only to itself), dropping it is exact: the sampler then runs 2 branches,
~1/3 less compute, the same math as the 3-branch reference.

`solve_closed` runs in float64 in PyTorch (unsupported on Apple GPUs).  Its
coefficients depend only on the timestep schedule, so they are computed once in
float64 with numpy, and the per-step update is `x_next = a * x + b * v`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import mlx.core as mx

from .flow_transformer import FlowTransformerMLX


class FlowSamplerMLX:
    def __init__(self, model: FlowTransformerMLX, nfe: int = 10, a_cfg_scale: float = 2.0,
                 u_cfg_scale: float = 0.0, num_train_timestep: int = 1000,
                 skip_noop_steps: bool = True):
        self.model = model
        self.a_cfg_scale = a_cfg_scale
        self.u_cfg_scale = u_cfg_scale
        self.clip = model.num_frames_for_clip          # 50
        self.block = model.num_frames_per_block        # 10
        self.ctx = 2                                   # overlap tokens per block
        self.dim_w = model.opt.dim_w

        # AvatarForcing: denoising_step_list = linspace(num_train_timestep, 0, nfe - 1)
        self.timesteps = np.linspace(num_train_timestep, 0, nfe - 1).tolist()
        # FlowMatchScheduler(shift=1, sigma_min=0, extra_one_step=False).set_timesteps(1000)
        self._sched_sigmas = np.linspace(1.0, 0.0, num_train_timestep)
        self._sched_timesteps = self._sched_sigmas * num_train_timestep

        self._coef_cache: dict = {}
        self._adaLN_cache: dict = {}
        self.kv_cache: Optional[list] = None
        self.branches = 2

        # A step whose current and next timestep are both 0 leaves every token
        # unchanged (solve_closed's hist_mask), so its forward pass is wasted.
        self.steps = [i for i in range(len(self.timesteps))
                      if not (skip_noop_steps and self.timesteps[i] == 0 and self._next_t(i) == 0)]

    # ---------------- schedule ----------------

    def _next_t(self, i: int) -> float:
        return self.timesteps[i + 1] if i < len(self.timesteps) - 1 else self.timesteps[i]

    def _sigma(self, t: np.ndarray) -> np.ndarray:
        idx = np.abs(self._sched_timesteps[None, :] - t[:, None]).argmin(axis=-1)
        return np.where(t == 0, 0.0, self._sched_sigmas[idx])

    def _coefs(self, i: int, n_ctx: int, n_new: int):
        """(a, b) of shape (1, n_ctx + n_new, 1) with x_next = a * x + b * v."""
        key = (i, n_ctx, n_new)
        if key not in self._coef_cache:
            t = np.array([0.0] * n_ctx + [self.timesteps[i]] * n_new, dtype=np.float64)
            tn = np.array([0.0] * n_ctx + [self._next_t(i)] * n_new, dtype=np.float64)
            s_t, s_tp = self._sigma(t), self._sigma(tn)
            ratio = np.where(s_t > 0, s_tp / np.maximum(s_t, 1e-8), 0.0)
            c_x0 = (1.0 - s_tp) - ratio * (1.0 - s_t)
            a, b = ratio + c_x0, -c_x0 * s_t
            hist = (t == 0) & (tn == 0)
            a, b = np.where(hist, 1.0, a), np.where(hist, 0.0, b)
            self._coef_cache[key] = (mx.array(a.reshape(1, -1, 1), dtype=mx.float32),
                                     mx.array(b.reshape(1, -1, 1), dtype=mx.float32))
        return self._coef_cache[key]

    # ---------------- conditioning ----------------

    def _adaLN(self, batch: int, seq_len: int, context_len: int):
        key = (batch, seq_len, context_len)
        if key not in self._adaLN_cache:
            m = self.model
            steps = m.precompute_timestep_adaLN(self.timesteps, batch, seq_len, context_len)
            zero = m.precompute_zero_adaLN(batch, seq_len + context_len)
            mx.eval(steps, zero)
            self._adaLN_cache[key] = (steps, zero)
        return self._adaLN_cache[key]

    def prepare_cfg_condition(self, wa: mx.array, wr: mx.array, seq_len: int,
                              user_wa: Optional[mx.array] = None, user_motion: Optional[mx.array] = None):
        """AvatarForcing.prepare_cfg_condition.  Branch order [uncond, audio, user];
        the user branch is dropped when u_cfg == 0 (see module docstring)."""
        null = mx.zeros_like(wa)
        if self.branches == 3:
            uw = user_wa if user_wa is not None else null
            um = user_motion if user_motion is not None else null
            wa_cat = mx.concatenate([null, wa, null], axis=0)
            user_wa_cat = mx.concatenate([null, null, uw], axis=0)
            motion_cat = mx.concatenate([null, null, um], axis=0)
        else:
            wa_cat = mx.concatenate([null, wa], axis=0)
            user_wa_cat = motion_cat = mx.concatenate([null, null], axis=0)
        c = self.model.compute_condition(wa_cat, user_wa_cat, motion_cat)
        wr_emb = self.model.compute_wr_embedded(mx.concatenate([wr] * self.branches, axis=0), seq_len)
        return c, wr_emb

    # ---------------- loop ----------------

    def reset(self, batch_size: int = 1, with_user: bool = False):
        """AvatarForcing.initialize_kv_cache (one cache per CFG branch).  `with_user`:
        user audio/motion will be passed; the 3rd branch is used only if u_cfg != 0."""
        self.branches = 3 if (with_user and self.u_cfg_scale != 0.0) else 2
        self.kv_cache = self.model.init_kv_cache(batch_size * self.branches)

    def _combine(self, v: mx.array) -> mx.array:
        parts = mx.split(v, self.branches, axis=0)
        v_t = parts[0] + self.a_cfg_scale * (parts[1] - parts[0])
        if self.branches == 3:
            v_t = v_t + self.u_cfg_scale * (parts[2] - parts[0])
        return v_t

    def _denoise(self, x_t, c, wr_emb, adaLN_steps, start_pos, kv, n_ctx, n_new):
        for i in self.steps:
            x_cat = mx.concatenate([x_t] * self.branches, axis=0)
            v = self.model.forward_with_precomputed(x_cat, c, adaLN_steps[i], wr_emb, start_pos,
                                                    kv_cache_list=kv, update_kv_cache=False)
            a, b = self._coefs(i, n_ctx, n_new)
            x_t = a * x_t + b * self._combine(v)
        return x_t

    def _update_kv_cache(self, x_t, c, wr_emb, adaLN_zero, start_pos):
        x_cat = mx.concatenate([x_t] * self.branches, axis=0)
        self.model.forward_with_precomputed(x_cat, c, adaLN_zero, wr_emb, start_pos,
                                            kv_cache_list=self.kv_cache, update_kv_cache=True)

    def prime(self, wa: mx.array, wr: mx.array, noise: mx.array,
              user_wa: Optional[mx.array] = None, user_motion: Optional[mx.array] = None) -> mx.array:
        """First block: 50 frames, no KV-cache read, then fill the cache.
        wa / user_wa / user_motion: (B, 50, dim_w); wr: (B, dim_w); noise: (B, 50, dim_w)."""
        B = wa.shape[0]
        if self.kv_cache is None:
            self.reset(B, with_user=user_wa is not None or user_motion is not None)
        c, wr_emb = self.prepare_cfg_condition(wa, wr, self.clip, user_wa, user_motion)
        steps, zero = self._adaLN(self.branches * B, self.clip, 0)
        x_t = self._denoise(noise.astype(mx.float32), c, wr_emb, steps, 0, None, 0, self.clip)
        self._update_kv_cache(x_t, c, wr_emb, zero, 0)
        return x_t

    def step(self, wa: mx.array, wr: mx.array, prev_x_t: mx.array, noise: mx.array, start_pos: int,
             user_wa: Optional[mx.array] = None, user_motion: Optional[mx.array] = None) -> mx.array:
        """Steady-state block.  wa / user_wa / user_motion: (B, 12, dim_w) = frames
        [t-2, t+10); prev_x_t: previous block's latents (last 2 tokens used);
        noise: (B, 10, dim_w); start_pos = t - 2.  Returns (B, 12, dim_w); the last 10
        are the new frames."""
        B = wa.shape[0]
        c, wr_emb = self.prepare_cfg_condition(wa, wr, self.block + self.ctx, user_wa, user_motion)
        steps, zero = self._adaLN(self.branches * B, self.block, self.ctx)
        x_t = mx.concatenate([prev_x_t[:, -self.ctx:], noise.astype(mx.float32)], axis=1)
        x_t = self._denoise(x_t, c, wr_emb, steps, start_pos, self.kv_cache, self.ctx, self.block)
        self._update_kv_cache(x_t, c, wr_emb, zero, start_pos)
        return x_t
