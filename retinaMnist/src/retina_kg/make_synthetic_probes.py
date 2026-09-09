"""Synthetic probe corpus with KNOWN ordinal structure, for integration testing.

Not a result and never written to results/. It exists so the whole retina path
-- feature build, adjacency, both heads, the controls, the ordinal metrics --
can be exercised deterministically before a single GPU-hour is spent, and so a
regression in any of them shows up as a number that stops making sense.

Generative model: finding f is present with high probability once the true grade
reaches f's ICDR level, low probability below it, plus noise. That is exactly
what the ontology claims, so a working pipeline must recover a high QWK and a
broken one must not.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import paths as P                                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(P.PACK))
    ap.add_argument("--npz-dir", default=str(P.NPZ))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="retina_p3")
    ap.add_argument("--noise", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fd = json.loads((Path(a.pack) / "findings.json").read_text())["findings"]
    names = [f["feature"] for f in fd]
    levels = np.array([f["level"] for f in fd])
    rng = np.random.default_rng(a.seed)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        y = np.load(Path(a.npz_dir) / f"{split}.npz")["labels"].reshape(-1)
        path = out / f"{a.tag}_{split}_0.jsonl"
        with path.open("w") as fh:
            for i, g in enumerate(y):
                present = (g >= levels).astype(float)
                p = 0.5 + 0.35 * (2 * present - 1)
                p = np.clip(p + rng.normal(0, a.noise, len(names)), 0.01, 0.99)
                fh.write(json.dumps({
                    "index": int(i), "phrasing": 3, "gold": int(g),
                    "p_yes": {n: round(float(v), 5) for n, v in zip(names, p)},
                }) + "\n")
        print(f"{split:5s} {len(y):5d} records -> {path.name}")


if __name__ == "__main__":
    main()
