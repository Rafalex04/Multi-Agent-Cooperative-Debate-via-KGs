"""BrEaST (BREAST-LESIONS-USG) -> npz, with radiologist descriptor ground truth.

This is the only public dataset that annotates the FINDINGS rather than only the
diagnosis, which is what makes E2 possible: every probe in this project has so
far been validated against malignancy, never against whether the finding it asks
about is actually present. Here it can be.

Framing follows the same registered policy as BUS-BRA -- a square window at 2.0x
the lesion box, shifted to stay in frame, resized to 224 -- so external numbers
across the two datasets are comparable. BrEaST ships a segmentation mask rather
than a bbox, so the box is the mask's extent.

Cases labelled `normal` carry no lesion and are dropped; that is the 256 -> 252
the dataset paper describes.
"""
from __future__ import annotations

import io, json, zipfile
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

# Which BI-RADS descriptor each KG finding is a question about, and when the
# radiologist's annotation counts as that finding being PRESENT. Recorded here
# rather than inferred, because the mapping is the experiment's main assumption.
# Three findings share the Halo column: they are three phrasings of one
# descriptor, and whether they agree with each other is itself a result.
DESCRIPTOR_MAP = {
    "irregular_shape":            ("Shape", lambda v: v == "irregular"),
    "oval_shape":                 ("Shape", lambda v: v in ("oval", "round")),
    "spiculated":                 ("Margin", lambda v: "spiculated" in v),
    "circumscribed_margin":       ("Margin", lambda v: v == "circumscribed"),
    "spiculated_or_irregular_mass": ("Margin+Shape", None),      # handled specially
    "posterior_shadowing_with_solid_irregular_mass":
                                  ("Posterior_features", lambda v: v in ("shadowing", "combined")),
    "anechoic_content":           ("Echogenicity", lambda v: v == "anechoic"),
    "hyperechoic_mass":           ("Echogenicity", lambda v: v == "hyperechoic"),
    "clustered_microcysts":       ("Echogenicity", lambda v: v == "complex cystic/solid"),
    "microcalcifications_in_hypoechoic_mass":
                                  ("Calcifications", lambda v: v in ("in a mass", "intraductal")),
    "echogenic_rind":             ("Halo", lambda v: v == "yes"),
    "echogenic_pseudocapsule":    ("Halo", lambda v: v == "yes"),
    "thin_uniform_pseudocapsule": ("Halo", lambda v: v == "yes"),
}
# No BrEaST column describes orientation or architectural distortion, so
# `parallel_orientation`, `non_parallel_orientation` and `architectural_distortion`
# have no ground truth here and are excluded from E2 rather than mapped loosely.


def _window(size, box, pad):
    W, H = size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    s = max(w, h) * pad / 2.0
    l, t, r, b = int(round(cx - s)), int(round(cy - s)), int(round(cx + s)), int(round(cy + s))
    if l < 0:
        r -= l; l = 0
    if t < 0:
        b -= t; t = 0
    if r > W:
        l -= (r - W); r = W
    if b > H:
        t -= (b - H); b = H
    return max(0, l), max(0, t), min(W, r), min(H, b)


def main(size=224):
    z = zipfile.ZipFile(_ZIP)
    df = pd.read_excel(_XLS)
    df = df[df["Classification"].isin(["benign", "malignant"])].reset_index(drop=True)

    imgs, labels, keep, desc_rows = [], [], [], []
    for _, r in df.iterrows():
        mfn = r["Mask_tumor_filename"]
        if not isinstance(mfn, str) or not mfn.strip():
            continue
        try:
            im = Image.open(io.BytesIO(z.read(_PRE + r["Image_filename"]))).convert("L")
            mk = np.asarray(Image.open(io.BytesIO(z.read(_PRE + mfn.split("&")[0]))).convert("L"))
        except KeyError:
            continue
        ys, xs = np.nonzero(mk > 0)
        if len(xs) == 0:
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        im = im.crop(_window(im.size, box, PAD)).resize((size, size), Image.BICUBIC)
        imgs.append(np.asarray(im, dtype=np.uint8))
        labels.append(MALIGNANT if r["Classification"] == "malignant" else BENIGN)
        keep.append(r["Image_filename"])
        desc_rows.append(r)

    imgs = np.array(imgs, dtype=np.uint8)
    labels = np.array(labels, dtype=np.uint8).reshape(-1, 1)

    # radiologist descriptor ground truth, one column per mappable finding
    D = pd.DataFrame(desc_rows).reset_index(drop=True)
    gt, cov = {}, {}
    for feat, (col, fn) in DESCRIPTOR_MAP.items():
        if col == "Margin+Shape":
            v = (D["Margin"].astype(str).str.contains("spiculated")
                 | (D["Shape"].astype(str) == "irregular"))
            na = ~(D["Margin"].astype(str).str.startswith("not applicable")
                   | D["Shape"].astype(str).str.startswith("not applicable"))
        else:
            s = D[col].astype(str)
            na = ~s.str.startswith("not applicable") & ~s.str.startswith("not available")
            v = s.map(fn)
        gt[feat] = np.where(na, v.astype(float), np.nan)
        cov[feat] = int(na.sum())

    out = _ROOT / "data/external/breast_pad2_224.npz"
    np.savez_compressed(
        out, imgs=imgs, labels=labels, ids=np.array(keep),
        birads=np.array([str(x) for x in D["BIRADS"]]),
        descriptor_names=np.array(list(gt.keys())),
        descriptors=np.array([gt[k] for k in gt]).T,
    )
    n_mal = int((labels.ravel() == MALIGNANT).sum())
    print(f"n={len(imgs)}  malignant={n_mal}  benign={len(imgs)-n_mal}  -> {out.name}")
    print("\ndescriptor ground truth (positives / annotated):")
    for k in gt:
        v = gt[k]
        print(f"  {k:46s} {int(np.nansum(v)):4d} / {cov[k]:4d}")
    (_HERE / "results/breast_prep.json").write_text(json.dumps(
        {"n": len(imgs), "n_malignant": n_mal,
         "descriptor_positives": {k: int(np.nansum(gt[k])) for k in gt},
         "descriptor_annotated": cov}, indent=1))


if __name__ == "__main__":
    main()
