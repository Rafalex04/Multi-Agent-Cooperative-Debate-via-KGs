"""BUS-BRA with the segmentation applied as a mask, per PREREGISTRATION_2026-08-27.

Framing is the registered 2.0x lesion-box window in every arm -- identical to
data/external/busbra_pad2_224.npz -- so the only thing that differs is what
happens to the pixels before the crop.

  bdim    outside the lesion attenuated to 35%.  Attenuated, not zeroed: a hard
          black surround is a synthetic feature present in no real ultrasound.
  bring   the lesion boundary drawn as a 2px contour, every pixel kept.

Exclusions are recorded and are applied identically across arms by construction:
a row that fails here fails in both mask arms, and the analysis intersects all
three arms on sample index before scoring.
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
PAD = 2.0
DIM = 0.35


def _window(size, bbox, pad):
    """Identical to prepare_busbra.py -- square, lesion-centred, shifted to stay in frame."""
    W, H = size
    x, y, w, h = (int(v) for v in bbox.strip("[]").split(","))
    cx, cy = x + w / 2.0, y + h / 2.0
    s = max(w, h) * pad / 2.0
    l, t, r, b = int(round(cx - s)), int(round(cy - s)), int(round(cx + s)), int(round(cy + s))
    if l < 0: r -= l; l = 0
    if t < 0: b -= t; t = 0
    if r > W: l -= (r - W); r = W
    if b > H: t -= (b - H); b = H
    return max(0, l), max(0, t), min(W, r), min(H, b)


def _ring(m, width=2):
    def sor(a):
        o = a.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o |= np.roll(np.roll(a, dy, 0), dx, 1)
        return o
    def sand(a):
        o = a.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            o &= np.roll(np.roll(a, dy, 0), dx, 1)
        return o
    d, e = m.copy(), m.copy()
    for _ in range(width):
        d = sor(d); e = sand(e)
    return d & ~e


def main(mode, size=224):
    z = zipfile.ZipFile(_ZIP)
    rows = list(csv.DictReader(io.StringIO(z.read("BUSBRA/bus_data.csv").decode("utf8"))))
    rows.sort(key=lambda r: r["ID"])                      # same order as the baseline npz

    imgs = np.zeros((len(rows), size, size), dtype=np.uint8)
    labels = np.zeros((len(rows), 1), dtype=np.uint8)
    ids, cases, birads, device, histo = [], [], [], [], []
    excluded = []

    for i, r in enumerate(rows):
        im = Image.open(io.BytesIO(z.read(f"BUSBRA/Images/{r['ID']}.png"))).convert("L")
        mfn = f"BUSBRA/Masks/{r['ID'].replace('bus_', 'mask_')}.png"
        why = None
        try:
            mk = np.asarray(Image.open(io.BytesIO(z.read(mfn))).convert("L")) > 0
        except KeyError:
            mk, why = None, "mask missing"
        if mk is not None and mk.shape != (im.size[1], im.size[0]):
            why = f"shape {mk.shape} vs image {im.size}"
        elif mk is not None and not mk.any():
            why = "mask empty"
        if why:
            excluded.append({"index": i, "id": r["ID"], "reason": why})
            mk = None

        a = np.asarray(im, dtype=np.float32)
        if mk is not None:
            a = np.where(mk, a, a * DIM) if mode == "bdim" else \
                np.where(_ring(mk), 255.0, a)
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        out = out.crop(_window(out.size, r["BBOX"], PAD)).resize((size, size), Image.BICUBIC)
        imgs[i] = np.asarray(out, dtype=np.uint8)
        labels[i, 0] = MALIGNANT if r["Pathology"] == "malignant" else BENIGN
        ids.append(r["ID"]); cases.append(int(r["Case"])); birads.append(int(r["BIRADS"]))
        device.append(r["Device"]); histo.append(r["Histology"])

    # the baseline npz is the authority on labels and case ids; assert we match it
    base = np.load(_ROOT / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    assert list(base["ids"]) == ids, "row order differs from the baseline npz"
    assert (base["labels"][:, 0] == labels[:, 0]).all(), "label mismatch vs baseline npz"
    assert (base["cases"] == np.array(cases, dtype=np.int32)).all(), "case mismatch"

    out = _ROOT / f"data/external/busbra_{mode}_{size}.npz"
    np.savez_compressed(out, imgs=imgs, labels=labels, ids=np.array(ids),
                        cases=np.array(cases, dtype=np.int32),
                        birads=np.array(birads, dtype=np.int8),
                        device=np.array(device), histology=np.array(histo))
    (_HERE / f"results/exclusions_{mode}.json").write_text(
        json.dumps({"n_rows": len(rows), "n_excluded": len(excluded),
                    "excluded": excluded}, indent=1))
    print(f"{mode}: n={len(rows)}  excluded={len(excluded)}  "
          f"malignant={int((labels[:,0]==MALIGNANT).sum())}  -> {out.name}")
    if excluded:
        for e in excluded[:10]:
            print(f"   EXCLUDED {e['id']}: {e['reason']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("bdim", "bring"))
    main(ap.parse_args().mode)
