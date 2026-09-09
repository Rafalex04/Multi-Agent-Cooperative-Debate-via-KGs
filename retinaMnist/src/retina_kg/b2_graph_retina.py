"""B2 GraphGeo's agent-collaboration graph, on retina.

One node per AGENT per image -- not per claim. That granularity difference is
what makes per-assertion ontology grounding impossible in GraphGeo's design and
is precisely what the comparison isolates.

Node features are the P3 probe vector, THE SAME visual input ThothGNN v3 gets, so
the comparison isolates the graph rather than the perception.

TWO EDGE RULES, both built, both reported. The smoke test showed all six
single-lineage agents returning the same binary stance at conf ~0.999, which
would make `r_conflict` fire never and render B2 a null BY CONSTRUCTION rather
than by measurement -- the exact failure the breast implementation warns about.
Rather than pick one rule silently:

  binary   agree if stance_i == stance_j          (faithful port of GraphGeo)
           conflict otherwise
  ordinal  agree if |mean_grade_i - mean_grade_j| <= delta
           conflict otherwise                     (uses the ordinal signal that
                                                   survives when stances collapse)

  transfer |conf_i - conf_j| > tau, directed high -> low, in both rules.

`edge_stats()` reports how often conflict actually fires, so the degeneracy is a
measured quantity in the results rather than an assumption.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P                                                    # noqa: E402
P.add_experiment_paths("26_baselines")
from corpus_io import load_corpus                                    # noqa: E402

SPLITS = ("train", "val", "test")
AGENTS = ("a0", "a1", "a2", "a3", "a4", "a5")
REL = ("agree", "conflict", "transfer")


def load_probe_split(tag, split, names, probe_dir):
    out = {}
    for f in sorted(Path(probe_dir).glob(f"{tag}_{split}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r.get("p_yes") or {}
            out[int(r["index"])] = np.array(
                [float(v[n]) if v.get(n) is not None else 0.5 for n in names])
    return out


def build(corpus, tau_transfer, rule="ordinal", delta=0.5, tag="retina_p3",
          splits=SPLITS, probe_dir=None, pack=None):
    """-> list of {x (6,F), edges {rel: (src,dst)}, y, split, sid, stances}"""
    pack = Path(pack) if pack else P.PACK
    probe_dir = Path(probe_dir) if probe_dir else P.RESULTS
    names = [f[0] for f in json.loads((pack / "probe_findings.json").read_text())]
    graphs = []
    by_split = load_corpus(corpus, splits=splits)
    for sp in splits:
        pr = load_probe_split(tag, sp, names, probe_dir)
        for d in by_split.get(sp, []):
            i = int(d["sample_id"])
            if i not in pr:
                continue
            st = d["stances"]
            if any(st.get(a, {}).get("stance") is None for a in AGENTS):
                continue
            img = pr[i]
            x = np.tile(img, (len(AGENTS), 1))
            stance = [st[a]["stance"] for a in AGENTS]
            conf = np.array([st[a]["conf"] for a in AGENTS], float)
            mg = np.array([st[a].get("mean_grade") if st[a].get("mean_grade")
                           is not None else 1.5 for a in AGENTS], float)
            src = {r: [] for r in REL}
            dst = {r: [] for r in REL}
            for u in range(len(AGENTS)):
                for v in range(len(AGENTS)):
                    if u == v:
                        continue
                    if rule == "binary":
                        same = stance[u] == stance[v]
                    else:
                        same = abs(mg[u] - mg[v]) <= delta
                    r = "agree" if same else "conflict"
                    src[r].append(v); dst[r].append(u)          # message v -> u
                    if conf[v] - conf[u] > tau_transfer:
                        src["transfer"].append(v); dst["transfer"].append(u)
            graphs.append({
                "sid": i, "split": sp, "y": float(d["gold_grade"]),
                "x": x.astype(np.float32),
                "edges": {r: (np.array(src[r], int), np.array(dst[r], int))
                          for r in REL},
                "stance": stance, "conf": conf, "mean_grade": mg,
                # `mal_share` is the key b2_model.collate reads for the anchor;
                # keeping the breast name means GraphGeo itself stays untouched.
                # The breast value is the malignant fraction over claims; the
                # retina analogue is the agents' mean claim grade normalised to
                # [0,1] by the top grade, which is continuous and carries more
                # than a referable-share fraction would (six agents that mostly
                # agree drive that to 0 or 1 and the anchor's logit saturates).
                "mal_share": float(np.clip(mg.mean() / 4.0, 1e-3, 1 - 1e-3)),
                "anchor": float(mg.mean()),
            })
    return graphs


def edge_stats(graphs):
    """How often each relation fires. A conflict rate of 0 means B2 is degenerate."""
    out = {}
    for r in REL:
        n = [len(g["edges"][r][0]) for g in graphs]
        out[r] = {"mean_per_graph": float(np.mean(n)) if n else 0.0,
                  "graphs_with_none": int(sum(1 for x in n if x == 0)),
                  "n_graphs": len(n)}
    return out


def summarise(graphs):
    by = {}
    for sp in SPLITS:
        g = [x for x in graphs if x["split"] == sp]
        if g:
            by[sp] = len(g)
    return by


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(P.PACK / "multi6"))
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.5)
    a = ap.parse_args()
    for rule in ("binary", "ordinal"):
        g = build(a.corpus, a.tau, rule=rule, delta=a.delta)
        if not g:
            print(f"{rule}: no graphs built"); continue
        es = edge_stats(g)
        print(f"\n{rule} rule  (tau={a.tau}, delta={a.delta})  splits {summarise(g)}")
        for r in REL:
            s = es[r]
            print(f"  {r:9s} {s['mean_per_graph']:5.2f} edges/graph   "
                  f"{s['graphs_with_none']}/{s['n_graphs']} graphs with none")
