"""The merged debate x knowledge graph, kept at claim resolution.

Every previous attempt in this project aggregated the debate into nine scalars per
finding node and ran the network on 16 nodes. That throws the debate graph away
before the network sees it: a mean over stances cannot express WHO spoke, WHEN, or
WHETHER THE CLAIM WAS REBUTTED, and it certainly cannot express that a rebuttal was
itself rebutted. Those are the only things a graph knows that a table does not.

So build the real thing, per sample:

    claim --addresses(AGREE:+1 / DISAGREE:-1)--> claim      the debate graph
    claim --cites-->                             finding     the merge
    finding --kg(+/- by stance agreement)-->     finding     the knowledge graph

The merge is structural, not feature-level: a claim node and a finding node are
joined by the citation edge the debate actually emitted, and `cited_features` is
drawn from the same all_findings() list the KG is indexed by, so the two graphs
share vertices rather than being two models averaged at the end.

Measured on debates_v5q (780 samples): 19.5 claims, 11.9 claim->claim edges and
19.5 citations per sample, no sample with zero edges, 2 agents, 3 rounds, and a
verdict distribution of 8134 DISAGREE against 1177 AGREE. The debate is mostly
adversarial, so the claim graph is mostly a rebuttal graph.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "15_kggnn"))
from claims import kg_findings                                        # noqa: E402
from kg_graph import build_adjacency                                  # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
SPLITS = ("train", "val", "test")
CMAX = 20                       # claims per sample are capped at 20 by the runner

# claim node features. Everything here is a property of the ARGUMENT, not of the
# image -- the image evidence enters through the finding nodes.
CLAIM_COLS = ("stance", "explicit", "r0", "r1", "r2", "repeat",
              "agent", "n_out", "in_agree", "in_disagree", "cites")


def _findings_index():
    f = kg_findings()
    return [n for n, _ in f], np.array([s for _, s in f], dtype=float)


def load_sample(path, names):
    """One debate -> (claim features, signed claim->claim, claim->finding, y)."""
    d = json.loads(Path(path).read_text())
    cl = d["claims"][:CMAX]
    pos = {c["node_id"]: i for i, c in enumerate(cl)}
    n = len(cl)

    Acc = np.zeros((CMAX, CMAX))         # signed: who rebutted / endorsed whom
    Acf = np.zeros((CMAX, len(names)))   # citation: the merge edge
    inA = np.zeros(CMAX); inD = np.zeros(CMAX)
    fidx = {nm: i for i, nm in enumerate(names)}

    for i, c in enumerate(cl):
        v = c.get("stance_verdict")
        sgn = 1.0 if v == "AGREE" else (-1.0 if v == "DISAGREE" else 0.0)
        for t in (c.get("addressed_ids") or []):
            j = pos.get(t)
            if j is None or sgn == 0.0:
                continue
            Acc[i, j] = sgn                       # i speaks about j
            (inA if sgn > 0 else inD)[j] += 1.0
        for f in (c.get("cited_features") or []):
            k = fidx.get(f)
            if k is not None:
                Acf[i, k] = 1.0

    X = np.zeros((CMAX, len(CLAIM_COLS)))
    for i, c in enumerate(cl):
        r = int(c.get("round_idx", 0))
        X[i] = [
            1.0 if c.get("label") == "MALIGNANT" else -1.0,
            1.0 if c.get("label_explicit") else 0.0,
            1.0 if r == 0 else 0.0, 1.0 if r == 1 else 0.0, 1.0 if r == 2 else 0.0,
            1.0 if c.get("is_repeat") else 0.0,
            1.0 if c.get("expert_id", "").endswith("1") else -1.0,
            float(len(c.get("addressed_ids") or [])),
            inA[i], inD[i], float(Acf[i].sum()),
        ]
    mask = np.zeros(CMAX); mask[:n] = 1.0
    y = 1.0 if d["gold_label"] == "MALIGNANT" else 0.0
    return X, Acc, Acf, mask, y, d["sample_id"]


def kg_operators(names, prior):
    """Signed KG adjacency over findings, split into agreeing and opposing parts."""
    A, _ = build_adjacency(_KG, names)
    same = (prior[:, None] * prior[None, :]) > 0
    return A * same, A * (~same)


def build(root, splits=SPLITS):
    """Load a whole debate corpus into padded arrays."""
    names, prior = _findings_index()
    out = {}
    for s in splits:
        files = sorted(glob.glob(f"{root}/{s}/*.json"))
        Xc, Acc, Acf, M, Y, ids = [], [], [], [], [], []
        for f in files:
            x, acc, acf, m, y, sid = load_sample(f, names)
            Xc.append(x); Acc.append(acc); Acf.append(acf); M.append(m); Y.append(y)
            ids.append(sid)
        out[s] = dict(Xc=np.array(Xc), Acc=np.array(Acc), Acf=np.array(Acf),
                      mask=np.array(M), y=np.array(Y), ids=ids)
    return out, names, prior


if __name__ == "__main__":
    root = str(_HERE.parents[2] / "breastMnist/data/breast/debates_v5q")
    d, names, prior = build(root)
    for s in SPLITS:
        g = d[s]
        print(f"{s:5s} n={len(g['y']):4d} claims/sample {g['mask'].sum(1).mean():5.2f} "
              f"cc-edges {(g['Acc'] != 0).sum(axis=(1, 2)).mean():5.2f} "
              f"cites {g['Acf'].sum(axis=(1, 2)).mean():5.2f} "
              f"mal {g['y'].mean():.3f}")
    Ap, An = kg_operators(names, prior)
    print(f"KG: {len(names)} findings, A+ {int((Ap > 0).sum())} edges, A- {int((An > 0).sum())} edges")
