# Adapted from AvatarForcing (https://github.com/TaekyungKi/AvatarForcing), commit 507e3046ad53:
# inference.py (DataProcessor.preprocess_face) and preprocess_user_video.py
# (UserVideoProcessor.dectect_and_crop_face_from_frames). Same detection and crop-box math,
# working on in-memory arrays instead of files. Copyright the AvatarForcing authors;
# adaptation by John Robinson. Licensed CC BY-NC 4.0 (see LICENSE and NOTICE).

"""Face cropping, as in AvatarForcing's preprocessing (needs the `crop` extra:
`face-alignment` + `opencv-python`; the S3FD detector runs on the CPU)."""
from __future__ import annotations

import numpy as np

_FA = None


def _detector():
    global _FA
    if _FA is None:
        import face_alignment
        _FA = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cpu")
    return _FA


def _detect(image: np.ndarray):
    """(box, mult) for the first face with score > 0.95, detected at 360 px height."""
    import cv2
    mult = 360.0 / image.shape[0]
    small = cv2.resize(image, dsize=(0, 0), fx=mult, fy=mult,
                       interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC)
    boxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), s)
             for (x1, y1, x2, y2, s) in _detector().face_detector.detect_from_image(small) if s > 0.95]
    if not boxes:
        raise ValueError("no face found (score > 0.95)")
    return boxes[0], mult


def _square_crop(h, w, bsy, bsx, my, mx, pad_ratio):
    bs = int(max(bsy, bsx) * (1 + pad_ratio))
    x1, y1, x2, y2 = mx - bs, my - bs, mx + bs, my + bs
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    bsx, bsy = x2 - x1, y2 - y1
    mx, my = int(x1 + bsx // 2), int(y1 + bsy // 2)
    return mx, my, int(min(bsx, bsy) // 2)


def crop_face(image: np.ndarray, size: int = 512, pad_ratio: float = 1.0) -> np.ndarray:
    """Avatar portrait: RGB uint8 [H, W, 3] -> face-centred square crop [size, size, 3]."""
    import cv2
    h, w = image.shape[:2]
    b, mult = _detect(image)
    mx, my, bs = _square_crop(h, w, int((b[3] - b[1]) / 2), int((b[2] - b[0]) / 2),
                              int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2), pad_ratio)
    face = image[my - bs: my + bs, mx - bs: mx + bs]
    return cv2.resize(face, dsize=(size, size), interpolation=cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC)


def crop_video(frames: np.ndarray, size: int = 512, pad_ratio: float = 1.0) -> np.ndarray:
    """User video: RGB uint8 [T, H, W, 3] -> one fixed crop (the mean face box over all
    frames) applied to every frame, [T, size, size, 3]."""
    import cv2
    h, w = frames.shape[1:3]
    stats = []
    for f in frames:
        b, mult = _detect(f)
        stats.append((int((b[3] - b[1]) / 2), int((b[2] - b[0]) / 2), int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2)))
    bsy, bsx, my, mx = np.mean(np.asarray(stats), axis=0)
    mx, my, bs = _square_crop(h, w, bsy, bsx, my, mx, pad_ratio)
    interp = cv2.INTER_AREA if mult < 1.0 else cv2.INTER_CUBIC
    return np.stack([cv2.resize(f[my - bs: my + bs, mx - bs: mx + bs], dsize=(size, size), interpolation=interp)
                     for f in frames])
