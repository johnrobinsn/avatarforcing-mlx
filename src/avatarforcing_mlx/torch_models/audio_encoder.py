# Vendored from TaekyungKi/AvatarForcing (https://github.com/TaekyungKi/AvatarForcing)
#   commit 507e3046ad53, file models/avatarforcing/AvatarForcing.py
# Copyright the AvatarForcing authors (Ki, Jang, Jo, Yoon, Hwang). Licensed CC BY-NC 4.0 (see LICENSE).
# Changes: excerpt — the AudioEncoder class only, with the imports it needs

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .wav2vec2 import Wav2VecModel


class AudioEncoder(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        # initialize audio encoder
        self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        self.wav2vec2.feature_extractor._freeze_parameters()
        for name, param in self.wav2vec2.named_parameters():
            param.requires_grad = False

        # set num of frames
        self.only_last_features = opt.only_last_features        
        self.num_frames_for_clip = int(opt.sec * self.opt.fps)
        self.samples_per_frame = int(opt.sampling_rate / opt.fps)

        # set audio projection layers
        audio_input_dim = 768 if opt.only_last_features else 12 * 768
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_input_dim, opt.dim_w),
            nn.LayerNorm(opt.dim_w),
            nn.SiLU())

    def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
        if self.only_last_features:
            a = a.last_hidden_state
        else:
            a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
            a = a.reshape(a.shape[0], a.shape[1], -1)
        return a

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if a.shape[1] % int(self.num_frames_for_clip * self.samples_per_frame) != 0:
            a = F.pad(a, (0, int(self.num_frames_for_clip * self.samples_per_frame) - a.shape[1]), mode='replicate')
        a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
        return self.audio_projection(a)

    @torch.inference_mode()
    def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
        if a.shape[1] % int(seq_len * self.samples_per_frame) != 0:
            a = F.pad(a, (0, int(seq_len * self.samples_per_frame) - a.shape[1]), mode='replicate')
        a = self.get_wav2vec2_feature(a, seq_len=seq_len)
        return self.audio_projection(a)
