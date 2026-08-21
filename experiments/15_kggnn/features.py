"""Per-sample node features for the 16 KG finding nodes.

This is where the knowledge graph and the debate graph actually meet. Each node
is a BI-RADS finding and carries three kinds of evidence about it:

  measured   p(yes) from a visual probe -- is the finding THERE
  argued     how the debate treated it -- net stance, volume, endorsement
  prior      what the KG says the finding means -- +1 malignant, -1 benign

The debate and the probe refer to the same finding because both are generated
from the same all_findings() list, so they can share a node rather than being
two separate models that have to be averaged at the end.
"""
from __future__ import annotations

import glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, by_split, kg_findings, load                # noqa: E402

COLS = ("p_yes", "measured", "net", "presence", "mass",
        "endorsed", "disputed", "open", "prior")


def probe_table(results_dir=None, tag="probe"):
    """(split, int index) -> {feature: p_yes}."""
    d = Path(results_dir or (_HERE.parent / "results"))
    out = {}
    for f in sorted(d.glob(f"{tag}_*.jsonl")):
        split = f.name.split("_")[1]
        for ln in f.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[(split, int(r["index"]))] = r["p_yes"]
    return out


def node_features(rec, findings, probes):
    """F x len(COLS) for one sample."""
    F = len(findings)
    names = [f for f, _ in findings]
    prior = [s for _, s in findings]
    X = np.zeros((F, len(COLS)))
    py = probes.get((rec["split"], int(rec["sid"])), {}) or {}

    grp = defaultdict(list)
    for c in rec["claims"]:
        if c["finding"] >= 0:
            grp[c["finding"]].append(c)
    tot = max(1, len(rec["claims"]))

    for i in range(F):
        cl = grp.get(i, [])
        n = len(cl)
        v = py.get(names[i])
        X[i] = [
            0.5 if v is None else v,                       # p_yes
            0.0 if v is None else 1.0,                     # measured
            (sum(1 if c["stance"] else -1 for c in cl) / n) if n else 0.0,
            1.0 if n else 0.0,
            n / tot,
            (sum(1 for c in cl if c["in_agree"] > 0) / n) if n else 0.0,
            (sum(1 for c in cl if c["in_disagree"] > 0) / n) if n else 0.0,
            (sum(1 for c in cl if c["round"] == 0) / n) if n else 0.0,
            prior[i],
        ]
    return X


def build(debate_roots, probe_dir=None, tag="probe"):
    """(by_split rows, X dict of (n, F, C) arrays, y dict) aligned on samples
    that have BOTH a debate and a probe vector."""
    findings = kg_findings()
    by = by_split(load(debate_roots, findings))
    probes = probe_table(probe_dir, tag)
    X, Y, rows = {}, {}, {}
    for s in SPLITS:
        keep = [r for r in by[s] if (s, int(r["sid"])) in probes]
        rows[s] = keep
        X[s] = np.array([node_features(r, findings, probes) for r in keep]) \
            if keep else np.zeros((0, len(findings), len(COLS)))
        Y[s] = np.array([r["y"] for r in keep], dtype=float)
    return rows, X, Y, findings
