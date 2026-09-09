"""Node tensor for the derma findings: SEEN, ARGUED, and what the ontology SAYS.

Same (n, F, d) contract as retina so the model code is untouched, but the third
channel differs because the label space is nominal:

    [probe p(present),  argument mass,  class concordance,  prior specificity]

  * argument mass    how many claims cited this finding (sum, not mean -- mean
                     cannot tell one finding argued heavily from many argued
                     lightly, which is the whole content of the channel)
  * class concordance of the claims citing this finding, the fraction that named
                     the class the ONTOLOGY assigns it. On retina this slot held
                     a signed "net stance"; with seven unordered classes there is
                     no direction to lean in, so the question becomes whether the
                     debate agrees with the ontology about what the finding means.
  * prior specificity 1 / (classes the finding supports). A finding pointing at
                     one class is worth more than one pointing at three.

The full 7-vector prior does not enter here -- it is the L2 penalty centre of the
class-specific readout in nominal.py, which is where class identity belongs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CMAX = 24


def load_findings(pack_dir):
    d = json.loads((Path(pack_dir) / "findings.json").read_text())
    f = d["findings"]
    return ([x["feature"] for x in f],
            np.array([x["prior"] for x in f], dtype=float),      # F x C
            [x["classes"] for x in f], d)


def probe_block(names, probe_dir, tag, splits=("train", "val", "test")):
    out = {}
    for f in sorted(Path(probe_dir).glob(f"{tag}_*.jsonl")):
        stem = f.name[len(tag) + 1:].rsplit("_", 1)[0]
        if stem not in splits:
            continue
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            v = r.get("p_yes") or {}
            out[(stem, int(r["index"]))] = np.array(
                [float(v[n]) if v.get(n) is not None else 0.5 for n in names])
    return out


def load_labels(npz_dir, splits=("train", "val", "test")):
    return {s: np.load(Path(npz_dir) / f"{s}.npz")["labels"].reshape(-1).astype(int)
            for s in splits}


def load_debate(root, names, classes, splits=("train", "val", "test")):
    """(split, index) -> (mass[F], concordance[F])."""
    fidx = {n: i for i, n in enumerate(names)}
    own = {n: set(c) for n, c in zip(names, classes)}
    out = {}
    root = Path(root)
    for s in splits:
        for f in sorted(root.glob(f"{s}_*.jsonl")):
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                mass = np.zeros(len(names)); agree = np.zeros(len(names))
                for c in (d.get("claims") or [])[:CMAX]:
                    lab = c.get("label")
                    for feat in (c.get("cited_features") or []):
                        k = fidx.get(feat)
                        if k is None:
                            continue
                        mass[k] += 1.0
                        if lab and lab in own.get(feat, ()):
                            agree[k] += 1.0
                conc = np.divide(agree, mass, out=np.zeros_like(mass), where=mass > 0)
                out[(s, int(d["sample_id"]))] = (mass, conc)
    return out


def build(pack_dir, npz_dir, probe_dir, tag, debate_root=None,
          splits=("train", "val", "test")):
    names, prior, classes, meta = load_findings(pack_dir)
    probes = probe_block(names, probe_dir, tag, splits)
    labels = load_labels(npz_dir, splits)
    deb = load_debate(debate_root, names, classes, splits) if debate_root else {}
    spec = np.array([1.0 / max(1, len(c)) for c in classes])

    F = len(names)
    X, Y, IDS = {}, {}, {}
    for s in splits:
        idxs = sorted(i for (sp, i) in probes if sp == s)
        A = np.zeros((len(idxs), F, 4))
        for r, i in enumerate(idxs):
            A[r, :, 0] = probes[(s, i)]
            m, c = deb.get((s, i), (None, None))
            if m is not None:
                A[r, :, 1] = m; A[r, :, 2] = c
        A[:, :, 3] = spec[None, :]
        X[s], Y[s], IDS[s] = A, labels[s][idxs], idxs
    return X, Y, IDS, names, prior, classes, meta


def standardise(X, splits=("train", "val", "test")):
    tr = X["train"].reshape(-1, X["train"].shape[-1])
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return {s: (X[s] - mu) / sd for s in splits}
