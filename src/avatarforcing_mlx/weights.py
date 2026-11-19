"""Model files: download the original checkpoints, convert the flow transformer for MLX.

Nothing is redistributed by this package.  `download()` fetches the files from where
their authors publish them:

  - AvatarForcing checkpoints (flow_transformer.pth, motion_autoencoder.pth) from the
    AvatarForcing authors' Google Drive (as in their download_weights.sh)
  - facebook/wav2vec2-base-960h from Hugging Face

and `convert()` writes flow_transformer_mlx.safetensors next to them.

Files live in $AVATARFORCING_MLX_HOME (default ~/.cache/avatarforcing-mlx).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

# Google Drive file ids from AvatarForcing's download_weights.sh (commit 507e3046ad53).
GDRIVE_IDS = {
    "motion_autoencoder.pth": "1a0ZqX9GMrkb3JFQgOBmUB0V-Bpwtv4Hp",
    "flow_transformer.pth": "1lchvHiDka0bBQ29bdY6lHxCyNGbGzvS_",
}
WAV2VEC2_REPO = "facebook/wav2vec2-base-960h"
WAV2VEC2_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]
MLX_WEIGHTS = "flow_transformer_mlx.safetensors"


def home() -> Path:
    return Path(os.environ.get("AVATARFORCING_MLX_HOME", Path.home() / ".cache" / "avatarforcing-mlx"))


def paths(root: Path | None = None) -> dict:
    root = Path(root) if root else home()
    return {
        "root": root,
        "flow_transformer": root / "flow_transformer.pth",
        "motion_autoencoder": root / "motion_autoencoder.pth",
        "wav2vec2": root / "wav2vec2-base-960h",
        "mlx": root / MLX_WEIGHTS,
    }


def missing(root: Path | None = None) -> list[str]:
    p = paths(root)
    out = [k for k in ("flow_transformer", "motion_autoencoder", "mlx") if not p[k].exists()]
    if not (p["wav2vec2"] / "config.json").exists():
        out.append("wav2vec2")
    return out


def download(root: Path | None = None, force: bool = False) -> dict:
    """Fetch the original checkpoints (if absent) and convert the flow transformer."""
    p = paths(root)
    p["root"].mkdir(parents=True, exist_ok=True)
    for name, file_id in GDRIVE_IDS.items():
        dst = p["root"] / name
        if dst.exists() and not force:
            continue
        import gdown
        print(f"downloading {name} (AvatarForcing authors' Google Drive) ...")
        out = gdown.download(id=file_id, output=str(dst), quiet=False)
        if out is None or not dst.exists():
            raise RuntimeError(
                f"could not download {name}. Google Drive may be rate-limiting; download it manually "
                f"from https://drive.google.com/drive/folders/1rN52J2QXD8A-r2CZ8nDqFdwvZuPRmMsc "
                f"(linked from https://github.com/TaekyungKi/AvatarForcing) into {p['root']}")
    if not (p["wav2vec2"] / "config.json").exists() or force:
        from huggingface_hub import snapshot_download
        print(f"downloading {WAV2VEC2_REPO} (Hugging Face) ...")
        snapshot_download(WAV2VEC2_REPO, local_dir=str(p["wav2vec2"]), allow_patterns=WAV2VEC2_FILES)
    if not p["mlx"].exists() or force:
        convert(p["flow_transformer"], p["mlx"])
    return p


# ---------------- conversion ----------------

DROP = "__DROP__"

# The checkpoint is a full AvatarForcing state_dict; only keys under this prefix
# belong to the flow transformer (the rest is audio_encoder.audio_projection, which
# stays in torch).
PREFIX = "flow_transformer."

# Keys (after prefix stripping) that differ between the PyTorch and MLX module trees.
# The MLX tree mirrors nn.Sequential with plain lists, so everything else maps 1:1.
RENAME_MAP: dict[str, str] = {
    # Static mask buffers, rebuilt at init.  Note: the checkpoint's copies are the
    # *inverse* of the masks the code generates (a training-time convention); the
    # reference inference never loads them (it copies parameters only).
    "alignment_mask": DROP,
    "block_bidir_causal_lookahead_mask": DROP,
}

_MHA_PACKED = re.compile(r"^(.+)\.in_proj_weight$")


def _unpack_mha(packed_weight: np.ndarray, packed_bias: np.ndarray | None, embed_dim: int) -> dict:
    """Split nn.MultiheadAttention's packed in_proj (3*E, E) into MLX's separate
    query/key/value projections."""
    assert packed_weight.shape[0] == 3 * embed_dim, f"expected {3 * embed_dim}, got {packed_weight.shape}"
    out = {f"{n}_proj.weight": packed_weight[i * embed_dim:(i + 1) * embed_dim]
           for i, n in enumerate(("query", "key", "value"))}
    if packed_bias is not None:
        out.update({f"{n}_proj.bias": packed_bias[i * embed_dim:(i + 1) * embed_dim]
                    for i, n in enumerate(("query", "key", "value"))})
    return out


def convert(src, dst, embed_dim: int = 512) -> Path:
    """AvatarForcing flow_transformer.pth -> MLX safetensors (float32).

    Steps: keep keys under `flow_transformer.`, drop the mask buffers, split the
    packed nn.MultiheadAttention projections.  Values are unchanged.
    """
    import torch
    from safetensors.numpy import save_file
    src, dst = Path(src), Path(dst)
    print(f"converting {src.name} -> {dst.name} ...")
    sd = torch.load(src, map_location="cpu", weights_only=True)
    if any(k.startswith(PREFIX) for k in sd):
        sd = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    out: dict[str, np.ndarray] = {}
    mha = {m.group(1) for k in sd if (m := _MHA_PACKED.match(k))}
    for prefix in mha:
        pb = sd.get(f"{prefix}.in_proj_bias")
        for suffix, arr in _unpack_mha(sd[f"{prefix}.in_proj_weight"].numpy(),
                                       pb.numpy() if pb is not None else None, embed_dim).items():
            out[f"{prefix}.{suffix}"] = arr.astype(np.float32)
    for k, v in sd.items():
        if k.rsplit(".", 1)[0] in mha and k.rsplit(".", 1)[1] in ("in_proj_weight", "in_proj_bias"):
            continue
        new_k = RENAME_MAP.get(k, k)
        if new_k == DROP:
            continue
        out[new_k] = v.numpy().astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst), metadata={
        "source": "AvatarForcing flow_transformer.pth (https://github.com/TaekyungKi/AvatarForcing)",
        "license": "CC BY-NC 4.0", "converted_by": "avatarforcing-mlx convert"})
    print(f"  {len(out)} tensors, {dst.stat().st_size / 1e6:.0f} MB")
    return dst


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else None)
