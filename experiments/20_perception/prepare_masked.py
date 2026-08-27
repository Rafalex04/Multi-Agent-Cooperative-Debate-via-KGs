"""Use the segmentation mask AS A MASK, not just as a source of a bounding box.

The crop sweep tested "crop to the mask with a margin" -- the window is the mask's
extent scaled by 1.25 / 2.0 / 3.0 -- and framing turned out to change nothing.
But that only ever used the mask to decide WHERE to cut. It never used it to say
which pixels are lesion, and those are different interventions:

  maskdim   background outside the lesion attenuated to 35%. Removes surrounding
            tissue as a distractor, and incidentally removes most of the burned-in
            calipers and Portuguese annotation that survive into the crops.
            Attenuated rather than zeroed: a hard black surround is a synthetic
            feature that exists in no real ultrasound.
  maskring  the lesion boundary drawn as a bright 2px contour, every pixel kept.
            Visual prompting -- the model is told where to look without losing
            the context it needs to judge margin and posterior features.

Framing is held at the registered 2.0 in both, so the ONLY difference from the
baseline arm is the mask treatment.

Prediction, stated before running. The shape failure is negation handling and is
constant to three decimals across a 2.4x change of window, so nothing here should
move it. The Halo group -- echogenic_rind, echogenic_pseudocapsule,
thin_uniform_pseudocapsule, all at or below chance in BOTH wordings at ALL three
framings -- is the one failure a mask could plausibly fix, because a thin
echogenic rim IS the boundary and `maskring` draws exactly that boundary. If the
Halo probes move and the shape probes do not, the two failure modes are confirmed
distinct. If nothing moves, the mask adds nothing over its bounding box.
"""
from __future__ import annotations

import argparse, io, json, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_ZIP = _ROOT / "data/external/BrEaST_images.zip"
_XLS = _ROOT / "data/external/BrEaST_clinical.xlsx"
_PRE = "BrEaST-Lesions_USG-images_and_masks/"
MALIGNANT, BENIGN = 0, 1
PAD = 2.0
DIM = 0.35


def _window(size, box, pad):
    W, H = size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    s = max(w, h) * pad / 2.0
    l, t, r, b = int(round(cx - s)), int(round(cy - s)), int(round(cx + s)), int(round(cy + s))
    if l < 0: r -= l; l = 0
    if t < 0: b -= t; t = 0
    if r > W: l -= (r - W); r = W
    if b > H: t -= (b - H); b = H
    return max(0, l), max(0, t), min(W, r), min(H, b)


def _ring(m, width=2):
    """Boundary of a binary mask, by dilation minus erosion, without scipy."""
    def shift_or(a):
        o = a.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o |= np.roll(np.roll(a, dy, 0), dx, 1)
        return o
    def shift_and(a):
        o = a.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o &= np.roll(np.roll(a, dy, 0), dx, 1)
        return o
    d, e = m.copy(), m.copy()
    for _ in range(width):
        d = shift_or(d); e = shift_and(e)
    return d & ~e


def main(mode, size=224):
    z = zipfile.ZipFile(_ZIP)
    df = pd.read_excel(_XLS)
    df = df[df["Classification"].isin(["benign", "malignant"])].reset_index(drop=True)

    imgs, labels, keep = [], [], []
    for _, r in df.iterrows():
        mfn = r["Mask_tumor_filename"]
        if not isinstance(mfn, str) or not mfn.strip():
            continue
        try:
            im = Image.open(io.BytesIO(z.read(_PRE + r["Image_filename"]))).convert("L")
            mk = np.asarray(Image.open(io.BytesIO(
                z.read(_PRE + mfn.split("&")[0]))).convert("L")) > 0
        except KeyError:
            continue
        ys, xs = np.nonzero(mk)
        if len(xs) == 0:
            continue
        a = np.asarray(im, dtype=np.float32)
        if mode == "maskdim":
            a = np.where(mk, a, a * DIM)
        elif mode == "maskring":
            a = np.where(_ring(mk), 255.0, a)
        else:
            raise SystemExit(f"unknown mode {mode}")
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        out = out.crop(_window(out.size, box, PAD)).resize((size, size), Image.BICUBIC)
        imgs.append(np.asarray(out, dtype=np.uint8))
        labels.append(MALIGNANT if r["Classification"] == "malignant" else BENIGN)
        keep.append(r["Image_filename"])

    # descriptors and labels come straight from the registered npz, by id, so the
    # ground truth cannot drift between arms
    base = np.load(_ROOT / "data/external/breast_pad2_224.npz", allow_pickle=True)
    order = {i: k for k, i in enumerate(base["ids"])}
    idx = [order[i] for i in keep]
    assert list(base["labels"][idx, 0]) == labels, "label misalignment vs registered npz"

    out = _ROOT / f"data/external/breast_{mode}_{size}.npz"
    np.savez_compressed(
        out, imgs=np.array(imgs, dtype=np.uint8),
        labels=np.array(labels, dtype=np.uint8).reshape(-1, 1),
        ids=np.array(keep), birads=base["birads"][idx],
        descriptor_names=base["descriptor_names"],
        descriptors=base["descriptors"][idx])
    print(f"{mode}: n={len(imgs)}  malignant={sum(1 for l in labels if l == MALIGNANT)}"
          f"  -> {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("maskdim", "maskring"))
    a = ap.parse_args()
    main(a.mode)
