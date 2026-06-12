"""Load BreastMNIST samples as base64 JPEG dicts for the ablation pipeline.

Each returned sample is a dict with keys:
  id         — zero-padded test-set index string, e.g. "009"
  label      — "BENIGN" or "MALIGNANT"
  image_b64  — base64-encoded RGB JPEG at cfg.data.image_size × cfg.data.image_size
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from PIL import Image

logger = logging.getLogger(__name__)

# MedMNIST v2 BreastMNIST label convention: 0 → MALIGNANT, 1 → BENIGN
_LABEL_MAP = {0: "MALIGNANT", 1: "BENIGN"}


def _image_to_b64(img_array: np.ndarray, size: int) -> str:
    """Convert a numpy image array to a base64-encoded RGB JPEG string."""
    img = Image.fromarray(img_array.astype(np.uint8))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_samples(cfg: DictConfig) -> list[dict]:
    """Load the fixed ablation sample set from BreastMNIST test split.

    Reads indices from cfg.data.indices_path, loads that many samples
    (up to cfg.data.num_samples), and returns them as base64 JPEG dicts.
    """
    from medmnist import BreastMNIST

    indices_path = Path(cfg.data.indices_path)
    if not indices_path.exists():
        raise FileNotFoundError(f"Ablation indices not found: {indices_path}")

    with indices_path.open() as f:
        raw = json.load(f)
    all_indices: list[int] = raw["indices"] if isinstance(raw, dict) else raw

    num_samples = int(cfg.data.num_samples)
    selected = all_indices[:num_samples]
    image_size = int(cfg.data.image_size)

    logger.info(
        "Loading %d BreastMNIST test samples (indices: %s ...)",
        num_samples,
        selected[:5],
    )

    dataset = BreastMNIST(split="test", download=True)
    images = dataset.imgs       # shape (N, H, W) or (N, H, W, C)
    labels = dataset.labels     # shape (N, 1)

    samples: list[dict] = []
    for idx in selected:
        raw = images[idx]
        # BreastMNIST images are grayscale (H, W) — PIL handles conversion to RGB.
        label_int = int(labels[idx][0])
        label_str = _LABEL_MAP.get(label_int, "BENIGN")
        image_b64 = _image_to_b64(raw, image_size)
        samples.append({
            "id": f"{idx:03d}",
            "label": label_str,
            "image_b64": image_b64,
        })
        logger.debug("Loaded sample idx=%d label=%s", idx, label_str)

    logger.info("Loaded %d samples", len(samples))
    return samples
