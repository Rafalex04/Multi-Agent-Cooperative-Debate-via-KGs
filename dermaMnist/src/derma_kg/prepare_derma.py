"""DermaMNIST -> the npz layout every runner in this project reads.

Same `imgs`/`labels` contract as breast and retina, so `run_probes.py` and the
debate runners consume it through their existing `--npz` flag unchanged.

CACHE DEFAULTS INSIDE THE REPO but is fully overridable with `--medmnist-root`
or `$DERMA_MEDMNIST_CACHE`. On the original cluster this pointed at node-local
/data instead of NFS home: DermaMNIST-224 is a 1,041 MB download and ~1.5 GB
uncompressed, against ~1.4 GB of home quota headroom there. The cache npz is
deleted after the split files are written and verified, since it is
redownloadable and the derived files are what everything reads -- so the
default here costs at most a transient 1.5 GB during preparation, not steady
state.

SPEC.md Key Design Decision 6 applies unchanged: `size=224` is passed explicitly
and a wrong shape is a hard exit.

Labels are the published 7-class DermaMNIST index 0..6, NOMINAL not ordinal:
  0 akiec  1 bcc  2 bkl  3 df  4 mel  5 nv  6 vasc
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402

SPLITS = ("train", "val", "test")
CACHE = Path(os.environ.get("DERMA_MEDMNIST_CACHE",
                            str(Path(__file__).resolve().parents[2] / "data/derma/.medmnist_cache")))


def _alive(lock):
    try:
        os.kill(int(lock.read_text().strip()), 0); return True
    except Exception:
        return False


def _sha256(path, blocks=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(blocks), b""):
            h.update(chunk)
    return h.hexdigest()


def export(out_dir, size=224, root=None, drop_cache=True, only=None):
    """`only` restricts to named splits, so a single corrupt file can be redone
    without refetching the rest.

    A LOCKFILE guards the whole export. Two concurrent processes writing the same
    .npz produced a 254 MB truncated train.npz that numpy reported as "not a zip
    file" -- silent until something tried to read it.
    """
    from medmnist import INFO, DermaMNIST
    lock = Path(out_dir).parent / ".export.lock"
    if lock.exists() and _alive(lock):
        raise SystemExit(f"another export is live (pid {lock.read_text().strip()})")

    root = Path(root) if root else CACHE
    root.mkdir(parents=True, exist_ok=True)
    info = INFO["dermamnist"]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"dataset": "dermamnist", "size": size, "task": info["task"],
           "n_channels": info["n_channels"], "label": info["label"],
           "medmnist_root": str(root), "out_dir": str(out_dir), "splits": {}}

    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    try:
      for sp in (only or SPLITS):
        ds = DermaMNIST(split=sp, download=True, size=size, root=str(root))
        imgs = np.asarray(ds.imgs, dtype=np.uint8)
        labels = np.asarray(ds.labels, dtype=np.int64).reshape(-1, 1)
        if imgs.shape[1:3] != (size, size):
            raise SystemExit(f"{sp}: got {imgs.shape[1:3]} not ({size},{size}) -- "
                             "medmnist ignored `size=`; refusing upscaled thumbnails.")
        if len(imgs) != info["n_samples"][sp]:
            raise SystemExit(f"{sp}: {len(imgs)} images, published "
                             f"{info['n_samples'][sp]}")
        dest = out_dir / f"{sp}.npz"
        np.savez_compressed(dest, imgs=imgs, labels=labels)
        counts = np.bincount(labels.ravel(), minlength=7).tolist()
        rec["splits"][sp] = {"n": int(len(imgs)), "shape": list(imgs.shape),
                             "class_counts": counts,
                             "class_share": [round(c / len(imgs), 4) for c in counts],
                             "bytes": dest.stat().st_size, "sha256": _sha256(dest)}
        print(f"{sp:5s} {imgs.shape}  classes {counts}  "
              f"{dest.stat().st_size/1e6:.0f} MB")
        del imgs, labels, ds

      if drop_cache:
        for f in root.glob("dermamnist*.npz"):
            n = f.stat().st_size
            f.unlink()
            print(f"  removed cache {f.name} ({n/1e6:.0f} MB) -- redownloadable")
    finally:
        lock.unlink(missing_ok=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(P.NPZ))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--medmnist-root", default=str(CACHE))
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of splits to (re)write")
    ap.add_argument("--results", default=str(P.RESULTS / "derma_prep.json"))
    a = ap.parse_args()
    rec = export(a.out_dir, a.size, a.medmnist_root, drop_cache=not a.keep_cache,
                 only=a.only.split(",") if a.only else None)
    out = Path(a.results); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(f"\nprovenance -> {out}")


if __name__ == "__main__":
    main()
