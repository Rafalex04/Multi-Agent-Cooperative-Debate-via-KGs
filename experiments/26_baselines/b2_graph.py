"""B2 - our implementation of GraphGeo's heterogeneous agent-collaboration graph.

G = (V, E, R) with R = {r_agree, r_conflict, r_transfer}. One node per AGENT per
image - not one per claim, which is the granularity difference that makes
per-assertion ontology grounding impossible in their design and is precisely
what the comparison is meant to isolate.

Edges come from PREDICTION GEOMETRY, not parsed verdicts. GraphGeo thresholds
geodesic distance between predicted coordinates; the binary-malignancy port is:

    r_agree     stance_i == stance_j
    r_conflict  stance_i != stance_j
    r_transfer  |conf_i - conf_j| > tau_transfer   (directed, high -> low)

r_transfer may coexist with agree or conflict, as in the paper.

Node features are [image_representation ; agent_embedding]. image_representation
is the P2+P3 probe vector - THE SAME visual input ThothGNN v3 receives - so the
comparison isolates the graph rather than the perception. The agent embedding is
learnable and lives in the model, not here.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
from claims import kg_findings                                   # noqa: E402
from corpus_io import load_corpus                                # noqa: E402

SPLITS = ("train", "val", "test")
AGENTS = ("a0", "a1", "a2", "a3", "a4", "a5")
_PROBES = _HERE.parents[0] / "15_kggnn/results"
_ROOT = _HERE.parents[1] / "breastMnist/data/breast"
REL = ("agree", "conflict", "transfer")


def load_probe_split(tag, split, names):
    out = {}
    for f in sorted(_PROBES.glob(f"{tag}_{split}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r.get("p_yes") or {}
            out[int(r["index"])] = np.array(
                [float(v[n]) if v.get(n) is not None else 0.5 for n in names])
    return out


def build(corpus, tau_transfer, phrasings=("probeneg", "probep3")):
    """-> list of dicts: {x (6,P), edges {rel: (src,dst) arrays}, y, split, sid}"""
    findings = kg_findings()
    names = [f for f, _ in findings]
    graphs = []
    corpus_by_split = load_corpus(corpus)
    for sp in SPLITS:
        pr = [load_probe_split(t, sp, names) for t in phrasings]
        for d in corpus_by_split.get(sp, []):
            i = int(d["sample_id"])
            if not all(i in p for p in pr):
                continue
            st = d["stances"]
            if any(st.get(a, {}).get("stance") is None for a in AGENTS):
                continue
            img = np.concatenate([p[i] for p in pr])          # 32-dim, P2+P3
            x = np.tile(img, (len(AGENTS), 1))                # same view per agent
            stance = [st[a]["stance"] for a in AGENTS]
            conf = np.array([st[a]["conf"] for a in AGENTS], float)
            src = {r: [] for r in REL}
            dst = {r: [] for r in REL}
            for u in range(len(AGENTS)):
                for v in range(len(AGENTS)):
                    if u == v:
                        continue
                    r = "agree" if stance[u] == stance[v] else "conflict"
                    src[r].append(v); dst[r].append(u)        # message v -> u
                    if conf[v] - conf[u] > tau_transfer:      # directed high->low
                        src["transfer"].append(v); dst["transfer"].append(u)
            graphs.append({
                "sid": d["sample_id"], "split": sp,
                "y": 1.0 if d["gold_label"] == "MALIGNANT" else 0.0,
                "x": x.astype(np.float32),
                "edges": {r: (np.array(src[r], int), np.array(dst[r], int))
                          for r in REL},
                "stance": stance, "conf": conf,
                "mal_share": _mal_share(d),
            })
    return graphs


def _mal_share(d):
    labs = [c.get("label") for c in d.get("claims", [])]
    labs = [l for l in labs if l in ("MALIGNANT", "BENIGN")]
    return sum(1 for l in labs if l == "MALIGNANT") / len(labs) if labs else 0.5


def roundtrip_test(graphs, n=25, seed=0):
    """Serialise -> deserialise -> compare. A graph builder that silently drops
    an edge type would make every downstream ablation meaningless."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=min(n, len(graphs)), replace=False)
    bad = 0
    for k in idx:
        g = graphs[k]
        blob = json.dumps({
            "sid": g["sid"], "split": g["split"], "y": g["y"],
            "x": g["x"].tolist(),
            "edges": {r: [g["edges"][r][0].tolist(), g["edges"][r][1].tolist()]
                      for r in REL},
            "stance": g["stance"], "conf": g["conf"].tolist(),
        })
        h = json.loads(blob)
        ok = (h["sid"] == g["sid"] and h["y"] == g["y"]
              and np.allclose(np.array(h["x"], np.float32), g["x"])
              and np.allclose(np.array(h["conf"]), g["conf"])
              and h["stance"] == g["stance"]
              and all(np.array_equal(np.array(h["edges"][r][0], int), g["edges"][r][0])
                      and np.array_equal(np.array(h["edges"][r][1], int), g["edges"][r][1])
                      for r in REL))
        bad += not ok
    return len(idx) - bad, len(idx)


def summarise(graphs):
    n = len(graphs)
    if not n:
        return "no graphs"
    per = {r: np.array([len(g["edges"][r][0]) for g in graphs]) for r in REL}
    nconf = sum(1 for g in graphs if len(set(g["stance"])) > 1)
    lines = [f"graphs {n}  malignant {int(sum(g['y'] for g in graphs))}",
             f"images with BOTH stances present: {nconf} ({nconf/n:.1%})"]
    for r in REL:
        lines.append(f"  r_{r:9s} mean {per[r].mean():5.2f} edges/graph   "
                     f"zero-edge graphs {int((per[r]==0).sum())} ({(per[r]==0).mean():.1%})")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(_ROOT / "multi6"))
    ap.add_argument("--tau", type=float, default=0.01)
    a = ap.parse_args()
    g = build(a.corpus, a.tau)
    print(summarise(g))
    ok, tot = roundtrip_test(g)
    print(f"\nround-trip serialisation: {ok}/{tot} graphs identical after "
          f"json encode/decode {'OK' if ok == tot else 'FAILED'}")
