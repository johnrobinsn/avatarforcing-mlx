"""Image / audio / video I/O (PyAV + Pillow; no ffmpeg binary needed)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

SAMPLE_RATE = 16000
FPS = 25


def load_image(path: str, size: int = 512) -> np.ndarray:
    """RGB uint8 [size, size, 3] (resized if needed; see preprocess.crop_face to crop)."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return np.array(img)


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 PCM at `sr`, from any file PyAV can read (wav, mp3, mp4, ...)."""
    import av
    with av.open(path) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks = []
        for frame in c.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in res.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in res.resample(None)]
    return np.concatenate(chunks).astype(np.float32)


def load_video_frames(path: str, fps: int = FPS) -> np.ndarray:
    """RGB uint8 [T, H, W, 3] at `fps` (nearest-frame resampling by timestamp)."""
    import av
    with av.open(path) as c:
        frames, times = [], []
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
            times.append(float(f.time) if f.time is not None else len(times) / fps)
    if not frames:
        raise ValueError(f"no video frames in {path}")
    times = np.asarray(times)
    n_out = int(np.floor((times[-1] - times[0]) * fps)) + 1
    idx = np.searchsorted(times, times[0] + np.arange(n_out) / fps, side="right") - 1
    return np.stack([frames[max(i, 0)] for i in idx])


def normalize_audio(pcm: np.ndarray) -> np.ndarray:
    """Wav2Vec2FeatureExtractor (do_normalize=True): zero mean, unit variance."""
    return ((pcm - pcm.mean()) / np.sqrt(pcm.var() + 1e-7)).astype(np.float32)


def to_uint8(pixels) -> np.ndarray:
    """torch [k, 3, H, W] in [-1, 1] -> numpy [k, H, W, 3] uint8."""
    import torch
    x = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def write_video(path: str, frames: np.ndarray, pcm: np.ndarray | None = None,
                fps: int = FPS, sr: int = SAMPLE_RATE, crf: int = 18) -> None:
    """H.264 mp4 from RGB uint8 frames [T, H, W, 3], with optional mono audio."""
    import av
    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        vs.options = {"crf": str(crf)}
        if pcm is not None:
            a_s = out.add_stream("aac", rate=sr)
            a_s.layout = "mono"
        for f in frames:
            for pkt in vs.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in vs.encode():
            out.mux(pkt)
        if pcm is not None:
            pcm = np.asarray(pcm, np.float32)[: int(len(frames) / fps * sr)]
            for i in range(0, len(pcm), 1024):
                af = av.AudioFrame.from_ndarray(pcm[None, i:i + 1024], format="flt", layout="mono")
                af.sample_rate, af.pts, af.time_base = sr, i, Fraction(1, sr)
                for pkt in a_s.encode(af):
                    out.mux(pkt)
            for pkt in a_s.encode():
                out.mux(pkt)
