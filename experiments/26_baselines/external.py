"""Step 8: BUS-BRA external evaluation for B1-full and B2-full.

Different country, four scanners, biopsy-proven. Folds are split on CASE so the
two views of a lesion cannot straddle a boundary, and every AUC is reported at
PATIENT level as well as image level.

Both operating points are evaluated where the data allows:
  auto  busbra_pad2_224.npz   (fully automatic, no segmentation)
  mask  busbra_bring_224.npz  (2.0x crop + maskring)

BreastMNIST has no segmentation masks - MedMNIST ships imgs and labels only -
so the mask condition exists for BUS-BRA alone.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in (".", "../16_external", "../17_hetgnn", "../14_kgtensor"):
    sys.path.insert(0, str((_HERE / _d).resolve()))
from analyse_e1 import auc, bacc, paired_bootstrap          # noqa: E402
from claims import kg_findings                              # noqa: E402
from corpus_io import load_corpus                           # noqa: E402
from powered_gnn import case_folds                          # noqa: E402

ARMS = {"auto": ("busbra_pad2_224.npz", ("busbra_p2", "busbra_p3"), "busbra"),
        "mask": ("busbra_bring_224.npz", ("bring_p2", "bring_p3"), "bring")}
EXT = _HERE.parents[0] / "16_external/results"
MASKRES = _HERE.parents[0] / "23_masktest/results"


def mal_share(ls):
    ls = [l for l in ls if l in ("MALIGNANT", "BENIGN")]
    return sum(1 for l in ls if l == "MALIGNANT") / len(ls) if ls else 0.5


def load_probes(tags, names):
    """index -> (F, len(tags)) matrix, only for samples measured in every arm."""
    per = {}
    for ti, t in enumerate(tags):
        d = MASKRES if t.startswith("bring") else EXT
        for f in sorted(Path(d).glob(f"{t}_*.jsonl")):
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                per.setdefault(int(r["index"]), {})[ti] = r.get("p_yes") or {}
    out = {}
    for i, d in per.items():
        if len(d) != len(tags):
            continue
        M = np.full((len(names), len(tags)), 0.5)
        for ti, v in d.items():
            for j, n in enumerate(names):
                if v.get(n) is not None:
                    M[j, ti] = float(v[n])
        out[i] = M
    return out


def to_case(v, cases, u):
    return np.array([v[cases == c].mean() for c in u])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="auto", choices=tuple(ARMS))
    ap.add_argument("--b1-corpus", required=True)
    ap.add_argument("--b2-corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()

    npz, tags, _ = ARMS[a.arm]
    names = [f for f, _ in kg_findings()]
    z = np.load(_HERE.parents[1] / "data/external" / npz, allow_pickle=True)
    y_all = (z["labels"][:, 0] == 0).astype(float)
    cases_all = z["cases"]
    probes = load_probes(tags, names)
    print(f"arm={a.arm}  npz={npz}  probes measured in both phrasings: {len(probes)}")

    res = {}
    # ---------------- B1 ----------------
    b1 = load_corpus(a.b1_corpus, splits=("all",))["all"]
    print(f"B1 corpus: {len(b1)} records")
    if b1:
        idx = [int(r["sample_id"]) for r in b1]
        y = y_all[idx]; cs = cases_all[idx]; u = np.unique(cs)
        yc = np.array([y[cs == c][0] for c in u])
        b1cfg = json.loads((_HERE / "results/b1.json").read_text())["frozen"]
        tone, tau = b1cfg["tone"], b1cfg["tau_conf"]
        s = []
        for r in b1:
            base = mal_share([c.get("label") for c in r["base_claims"]])
            br = r["branches"][tone]; mod = br.get("moderator") or {}
            full = (mal_share(list(mod.values())) if mod else
                    mal_share([c.get("label") for c in
                               r["base_claims"] + br["catfish"] + br["response"]]))
            t = r["trigger"]
            fired = t["silent_agreement"] or (t.get("mean_conf") is not None
                                              and tau is not None
                                              and t["mean_conf"] < tau)
            s.append(full if fired else base)
        s = np.array(s); sc = to_case(s, cs, u)
        res["B1 Catfish Agent (ours impl)"] = {
            "n_img": len(idx), "n_case": int(len(u)),
            "auc": float(auc(y, s)), "auc_case": float(auc(yc, sc)),
            "bacc_case": float(bacc(yc, sc, float(np.median(sc))))}
        print(f"  B1-full   image {auc(y,s):.4f}   patient {auc(yc,sc):.4f}")

    # ---------------- B2 ----------------
    from b2_graph import build
    from b2_train import train_eval
    g = build(a.b2_corpus, 0.01, phrasings=tags, splits=("all",),
              probes=probes)
    print(f"B2 graphs: {len(g)}")
    if len(g) > 200:
        gi = [int(x["sid"]) for x in g]
        y = y_all[gi]; cs = cases_all[gi]; u = np.unique(cs)
        yc = np.array([y[cs == c][0] for c in u])
        for lab, kw in (("B2 GraphGeo (ours impl, no KG)", {}),
                        ("B2 GraphGeo + our anchor", dict(anchored=True))):
            oof = np.zeros(len(g))
            for te in case_folds(cs, k=5, seed=0):
                tr = ~te
                oof[te] = np.mean([train_eval([x for x, m in zip(g, tr) if m],
                                              [x for x, m in zip(g, te) if m],
                                              sd, epochs=a.epochs, **kw)
                                   for sd in (0, 1, 2)], axis=0)
            sc = to_case(oof, cs, u)
            res[lab] = {"n_img": len(g), "n_case": int(len(u)),
                        "auc": float(auc(y, oof)), "auc_case": float(auc(yc, sc)),
                        "bacc_case": float(bacc(yc, sc, float(np.median(sc))))}
            print(f"  {lab:32s} image {auc(y,oof):.4f}   patient {auc(yc,sc):.4f}")

    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
