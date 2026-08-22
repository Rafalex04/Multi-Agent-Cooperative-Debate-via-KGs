"""BUS-BRA -> npz, in the exact shape the probe runner already expects.

Read straight out of the zip. The home filesystem is inode-limited, not
space-limited, and extracting 3750 PNGs would spend a quarter of the remaining
file budget on data that is only ever read as arrays.

FRAMING is the one real decision here, and it is made on appearance, never on
score. BreastMNIST is a BUSI-derived crop in which the lesion fills roughly half
the frame. BUS-BRA ships full panoramic sweeps in which the lesion is a small
part of a wide field, so feeding whole frames would measure my own preprocessing
rather than distribution shift. The other extreme is just as wrong: the supplied
BBOX is tight to the lesion, and cropping to it amputates the margin and removes
the posterior region entirely -- which would silently zero out `spiculated_margin`,
`circumscribed_margin` and `posterior_shadowing`, three of the sixteen findings.

So: a square window of side max(w,h)*PAD centred on the lesion. PAD=2.0 puts the
lesion at about half the frame, keeps the margin, and keeps roughly a lesion's
depth of tissue below it for posterior features. PAD=1.5 is retained as a
registered sensitivity arm.

Windows that run off the edge are SHIFTED back inside, never zero-padded; black
bars would be a synthetic feature that does not exist in BreastMNIST.

Known, measured domain artifact: BUS-BRA frames carry radiologist calipers and
burned-in Portuguese annotation. Saturated pixels survive into ~45% of PAD=2.0
crops, marginally MORE often in benign (46.8% vs 41.9%), so the artifact does not
leak malignancy and if anything cuts against the hypothesis under test.

Labels follow the BreastMNIST convention -- MALIGNANT=0, BENIGN=1 -- so that a
frozen model transfers without a sign flip.
"""
from __future__ import annotations

import argparse, csv, io, json, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_ZIP = _ROOT / "data/external/BUSBRA.zip"

MALIGNANT, BENIGN = 0, 1


def _window(size, bbox, pad):
    """Square lesion-centred window of side max(w,h)*pad, shifted to stay in frame."""
    W, H = size
    x, y, w, h = (int(v) for v in bbox.strip("[]").split(","))
    cx, cy = x + w / 2.0, y + h / 2.0
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


def build(pad, out, size=224):
    z = zipfile.ZipFile(_ZIP)
    rows = list(csv.DictReader(io.StringIO(z.read("BUSBRA/bus_data.csv").decode("utf8"))))
    rows.sort(key=lambda r: r["ID"])                      # deterministic order

    imgs = np.zeros((len(rows), size, size), dtype=np.uint8)
    labels = np.zeros((len(rows), 1), dtype=np.uint8)
    ids, cases, birads, device, histo = [], [], [], [], []

    for i, r in enumerate(rows):
        im = Image.open(io.BytesIO(z.read(f"BUSBRA/Images/{r['ID']}.png"))).convert("L")
        im = im.crop(_window(im.size, r["BBOX"], pad)).resize((size, size), Image.BICUBIC)
        imgs[i] = np.asarray(im, dtype=np.uint8)
        labels[i, 0] = MALIGNANT if r["Pathology"] == "malignant" else BENIGN
        ids.append(r["ID"]); cases.append(int(r["Case"])); birads.append(int(r["BIRADS"]))
        device.append(r["Device"]); histo.append(r["Histology"])

    np.savez_compressed(
        out, imgs=imgs, labels=labels,
        ids=np.array(ids), cases=np.array(cases, dtype=np.int32),
        birads=np.array(birads, dtype=np.int8),
        device=np.array(device), histology=np.array(histo),
    )
    n_mal = int((labels.ravel() == MALIGNANT).sum())
    print(f"pad={pad}  n={len(rows)}  malignant={n_mal} benign={len(rows)-n_mal} "
          f"cases={len(set(cases))}  -> {out.name} ({out.stat().st_size/1e6:.0f} MB)")
    return dict(pad=pad, n=len(rows), n_malignant=n_mal, n_cases=len(set(cases)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pads", default="2.0,1.5")
    a = p.parse_args()
    meta = {}
    for pad in [float(x) for x in a.pads.split(",")]:
        tag = f"{pad:g}".replace(".", "")
        meta[f"pad{tag}"] = build(pad, _ROOT / f"data/external/busbra_pad{tag}_224.npz")
    (_HERE / "results/busbra_prep.json").write_text(json.dumps(meta, indent=2))
