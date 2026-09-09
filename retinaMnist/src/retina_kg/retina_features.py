"""Node tensor for the retina findings: what was SEEN, ARGUED, and what ICDR SAYS.

Same three channels as `thothgnn3.node_tensor`, and the same shape contract
(n, F, d) so the model code is untouched:

    [probe arms...,  argument mass,  net stance,  ICDR level prior]

Two differences from breast, both in the label space rather than the structure:

  * labels are ICDR grades 0..4, read from the npz, not a two-name string map.
  * `net stance` is a claim's grade recentred on the referable-DR boundary,
    (grade - 1.5) / 2.5, the same transform the prior uses. On breast this
    column was +-1; here it is an ordinal position, so "which way the argument
    leans" keeps its meaning on a five-level scale.

The debate corpus is OPTIONAL. With none, mass and net are zero and the tensor
is the probe-plus-prior arm -- which is exactly v3's `no-debate` control, so the
perception-only result is available before any debate corpus exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CMAX = 20


def load_findings(pack_dir):
    d = json.loads((Path(pack_dir) / "findings.json").read_text())
    f = d["findings"]
    return ([x["feature"] for x in f],
            np.array([x["prior"] for x in f], dtype=float),
            np.array([x["level"] for x in f], dtype=float),
            d)


def probe_block(names, probe_dir, tags, splits=("train", "val", "test")):
    """(split, index) -> (F, n_tags) array of p(present), 0.5 where unmeasured."""
    out = {}
    for ti, t in enumerate(tags):
        for f in sorted(Path(probe_dir).glob(f"{t}_*.jsonl")):
            # <tag>_<split>_<shard>.jsonl -- the tag itself may contain '_'
            stem = f.name[len(t) + 1:].rsplit("_", 1)[0]
            if stem not in splits:
                continue
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                a = out.setdefault((stem, int(r["index"])),
                                   np.full((len(names), len(tags)), 0.5))
                for j, n in enumerate(names):
                    v = (r.get("p_yes") or {}).get(n)
                    if v is not None:
                        a[j, ti] = float(v)
    return out


def load_labels(npz_dir, splits=("train", "val", "test")):
    out = {}
    for s in splits:
        z = np.load(Path(npz_dir) / f"{s}.npz")
        out[s] = z["labels"].reshape(-1).astype(int)
    return out


def _stance(grade, centre=1.5, scale=2.5):
    return (float(grade) - centre) / scale


def load_debate(root, names, splits=("train", "val", "test")):
    """(split, index) -> (mass[F], net[F]) from claim->finding citations.

    Mirrors hetgraph.load_sample + thothgnn3.node_tensor, collapsed to the two
    columns the finding nodes actually consume. Sum aggregation, as on breast:
    mean cannot tell "one finding argued heavily" from "many argued lightly".
    """
    fidx = {n: i for i, n in enumerate(names)}
    out = {}
    root = Path(root)
    for s in splits:
        for f in sorted(root.glob(f"{s}/debate_*.json")) or sorted(root.glob(f"{s}_*.jsonl")):
            recs = ([json.loads(f.read_text())] if f.suffix == ".json"
                    else [json.loads(l) for l in f.read_text().splitlines() if l.strip()])
            for d in recs:
                mass = np.zeros(len(names)); net = np.zeros(len(names))
                for c in (d.get("claims") or [])[:CMAX]:
                    g = c.get("grade")
                    st = _stance(g) if g is not None else 0.0
                    for feat in (c.get("cited_features") or []):
                        k = fidx.get(feat)
                        if k is not None:
                            mass[k] += 1.0
                            net[k] += st
                out[(s, int(d["sample_id"]))] = (mass, net)
    return out


def build(pack_dir, npz_dir, probe_dir, tags, debate_root=None,
          splits=("train", "val", "test")):
    """-> X{split}, y{split}, ids{split}, names, prior, level."""
    names, prior, level, meta = load_findings(pack_dir)
    probes = probe_block(names, probe_dir, tags, splits)
    labels = load_labels(npz_dir, splits)
    deb = load_debate(debate_root, names, splits) if debate_root else {}

    F, npc = len(names), len(tags)
    X, Y, IDS = {}, {}, {}
    for s in splits:
        idxs = sorted(i for (sp, i) in probes if sp == s)
        if not idxs:
            X[s] = np.zeros((0, F, npc + 3)); Y[s] = np.zeros(0, int); IDS[s] = []
            continue
        A = np.zeros((len(idxs), F, npc + 3))
        for r, i in enumerate(idxs):
            A[r, :, 0:npc] = probes[(s, i)]
            m, n = deb.get((s, i), (None, None))
            if m is not None:
                A[r, :, npc] = m
                A[r, :, npc + 1] = n
        A[:, :, npc + 2] = prior[None, :]
        X[s], Y[s], IDS[s] = A, labels[s][idxs], idxs
    return X, Y, IDS, names, prior, level, meta


def standardise(X, splits=("train", "val", "test")):
    tr = X["train"].reshape(-1, X["train"].shape[-1])
    mu, sd = tr.mean(0), tr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return {s: (X[s] - mu) / sd for s in splits}
