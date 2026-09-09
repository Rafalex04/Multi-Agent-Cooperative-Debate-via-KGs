"""RetinaMNIST -> the npz layout every runner in this project already reads.

Writes `data/retina/images_224/{train,val,test}.npz` with the same `imgs` /
`labels` keys as `breastMnist/data/breast/images_224/*.npz`, so `run_probes.py`
and `run_debate_v5.py` consume it through their existing `--npz` flag with no
change. Two differences from the breast archives, both handled downstream:

  * imgs is (N, 224, 224, 3) -- RGB, not grayscale. Every consumer builds its
    base64 through `Image.fromarray(...).convert("RGB")`, so a colour array is
    already the fast path.
  * labels are 0..4 (ICDR severity), not the binary MALIGNANT=0/BENIGN=1
    convention. NOTHING here binarises: the grade is written as published and
    the reduction to referable DR is a modelling choice made at analysis time.

SPEC.md Key Design Decision 6 applies unchanged: `size=224` MUST be passed to
the constructor. medmnist >= 3.0 defaults `size=None` -> 28x28, and silently
upscaling 28px thumbnails is what destroyed the debate signal on breast.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402


def _sha256(path, blocks=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(blocks), b""):
            h.update(chunk)
    return h.hexdigest()


def export(out_dir, size=224, root=None):
    from medmnist import INFO, RetinaMNIST

    # medmnist >= 3.0 raises unless `root` already exists; it never creates it.
    root = Path(root) if root else Path.home() / ".medmnist"
    root.mkdir(parents=True, exist_ok=True)
    info = INFO["retinamnist"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"dataset": "retinamnist", "size": size, "task": info["task"],
           "medmnist_root": str(root),
           "n_channels": info["n_channels"], "label": info["label"], "splits": {}}

    for sp in SPLITS:
        ds = RetinaMNIST(split=sp, download=True, size=size, root=str(root))
        imgs = np.asarray(ds.imgs, dtype=np.uint8)
        labels = np.asarray(ds.labels, dtype=np.int64).reshape(-1, 1)
        if imgs.shape[1:3] != (size, size):
            raise SystemExit(
                f"{sp}: got {imgs.shape[1:3]} not ({size},{size}). "
                "medmnist ignored `size=` -- refusing to write upscaled thumbnails.")
        if len(imgs) != info["n_samples"][sp]:
            raise SystemExit(f"{sp}: {len(imgs)} images, published count "
                             f"{info['n_samples'][sp]}")
        dest = out_dir / f"{sp}.npz"
        np.savez_compressed(dest, imgs=imgs, labels=labels)
        counts = np.bincount(labels.ravel(), minlength=5).tolist()
        rec["splits"][sp] = {
            "n": int(len(imgs)), "shape": list(imgs.shape),
            "grade_counts": counts,
            "grade_share": [round(c / len(imgs), 4) for c in counts],
            "bytes": dest.stat().st_size, "sha256": _sha256(dest),
        }
        print(f"{sp:5s} {imgs.shape}  grades {counts}  "
              f"{dest.stat().st_size/1e6:.1f} MB")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(P.NPZ))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--medmnist-root", default=None,
                    help="cache dir for the downloaded npz (default ~/.medmnist)")
    ap.add_argument("--results", default=str(P.RESULTS / "retina_prep.json"))
    a = ap.parse_args()

    rec = export(a.out_dir, a.size, a.medmnist_root)
    out = Path(a.results); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(f"\nprovenance -> {out}")


if __name__ == "__main__":
    main()
