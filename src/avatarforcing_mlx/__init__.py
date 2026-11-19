"""AvatarForcing on Apple Silicon: an MLX port of the AvatarForcing flow transformer and
sampler, with the motion autoencoder and wav2vec2 on PyTorch MPS.

AvatarForcing: Ki, Jang, Jo, Yoon, Hwang, "Avatar Forcing: Real-Time Interactive Head
Avatar Generation for Natural Conversation", CVPR 2026 —
https://github.com/TaekyungKi/AvatarForcing.  Licensed CC BY-NC 4.0; see LICENSE and NOTICE.
"""
__version__ = "0.1.0"

from .flow_transformer import FlowTransformerMLX, load_weights
from .sampler import FlowSamplerMLX

__all__ = ["FlowTransformerMLX", "FlowSamplerMLX", "load_weights", "AvatarPipeline"]


def __getattr__(name):   # AvatarPipeline pulls in torch; import it lazily
    if name == "AvatarPipeline":
        from .pipeline import AvatarPipeline
        return AvatarPipeline
    raise AttributeError(name)
