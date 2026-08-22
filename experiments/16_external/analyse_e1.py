"""E1/E3/E4 -- apply the frozen models to external data and decide the round.

Nothing is fitted here. The weights, the standardisation and the bAcc thresholds
all come out of `frozen_models.json`, which was hashed before any external score
existed. Refitting appears once, clearly labelled as secondary, to separate "the
weights do not transfer" from "the probes do not transfer".

Endpoints, as registered:
  E1  external AUC/bAcc of kgsum and logit, image-level and case-level
      (case-level is primary; BUS-BRA has multiple views per case)
  E3  s5 vs lambda=0 vs 5 shuffled-KG draws, paired bootstrap, P(better) >= 0.95
  E4  Spearman rho against radiologist BI-RADS, and AUC for BI-RADS 4/5 vs 2/3
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
from claims import kg_findings                                        # noqa: E402
from kg_graph import build_adjacency                                  # noqa: E402
from s3_s5 import fit_calib                                           # noqa: E402

_KG = _HERE.parents[1] / "breastMnist/data/breast/knowledge_graph.json"
TAGS = ("busbra_p2", "busbra_p3")


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(kind="mergesort")
    r = np.empty(len(a)); sa = a[o]; i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def bacc(y, p, t):
    pred = np.asarray(p) >= t; y = np.asarray(y)
    tp = ((pred) & (y == 1)).sum(); fn = ((~pred) & (y == 1)).sum()
    tn = ((~pred) & (y == 0)).sum(); fp = ((pred) & (y == 0)).sum()
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))


def spearman(a, b):
    def rank(v):
        o = np.asarray(v, float).argsort(kind="mergesort")
        r = np.empty(len(v)); sv = np.asarray(v, float)[o]; i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_probes(names, tags, results_dir):
    """(index -> F x len(tags) probe matrix), only for fully-probed samples."""
    per = {}
    for ti, t in enumerate(tags):
        for f in sorted(Path(results_dir).glob(f"{t}_*.jsonl")):
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                d = per.setdefault(int(r["index"]), {})
                d[ti] = r["p_yes"] or {}
    out = {}
    for idx, d in per.items():
        if len(d) != len(tags):
            continue                                     # not yet measured in every arm
        M = np.full((len(names), len(tags)), 0.5)
        for ti, vals in d.items():
            for j, n in enumerate(names):
                v = vals.get(n)
                if v is not None:
                    M[j, ti] = float(v)
        out[idx] = M
    return out


def paired_bootstrap(y, a, b, n=4000, seed=0):
    """P(model a has higher AUC than b) over resampled subjects."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); wins = 0
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        wins += auc(y[i], a[i]) > auc(y[i], b[i])
    return wins / n


def main():
    F = json.loads((_HERE / "results/frozen_models.json").read_text())
    names = F["findings"]
    prior = np.array(F["prior"], float)
    mu, sd = np.array(F["mu"]), np.array(F["sd"])
    probes = load_probes(names, TAGS, _HERE / "results")
    z = np.load(_HERE.parents[1] / "data/external/busbra_pad2_224.npz")
    y_all = (z["labels"][:, 0] == 0).astype(int)          # 0 == MALIGNANT
    idx = sorted(probes)
    print(f"BUS-BRA: {len(idx)}/{len(y_all)} images fully probed in both arms")
    if len(idx) < 50:
        print("too few to analyse yet"); return
    X = np.array([probes[i].T.reshape(-1) for i in idx])  # tag-major, matches freeze
    y = y_all[idx]
    cases = z["cases"][idx]; birads = z["birads"][idx]
    Z = (X - mu) / sd
    print(f"  malignant {int(y.sum())}  benign {int((1-y).sum())}  cases {len(set(cases))}\n")

    res = {"n_images": len(idx), "n_cases": int(len(set(cases))), "models": {}}
    scores = {}
    print(f"{'model':10s} {'indom':>7s} {'IMAGE':>7s} {'bAcc':>7s} {'CASE':>7s} {'bAcc':>7s} {'gap':>8s}")
    for key in ("kgsum", "logit", "s5"):
        m = F["models"][key]
        if key == "s5":
            a, c, w = np.array(m["a"]), np.array(m["c"]), np.array(m["w"])
            ix = np.tile(np.arange(len(names)), len(TAGS))
            s = (Z * a[ix] + c[ix]) @ w + m["b"]
        else:
            s = Z @ np.array(m["w"]) + m["b"]
        thr = m["threshold"]
        u = np.unique(cases)
        sc_case = np.array([s[cases == cc].mean() for cc in u])
        y_case = np.array([y[cases == cc][0] for cc in u])
        r = {"auc_image": auc(y, s), "bacc_image": bacc(y, s, thr),
             "auc_case": auc(y_case, sc_case), "bacc_case": bacc(y_case, sc_case, thr),
             "indomain_test": m["indomain"]["test"]}
        r["gap"] = r["indomain_test"] - r["auc_case"]
        res["models"][key] = r
        scores[key] = (s, sc_case, y_case)
        print(f"{key:10s} {r['indomain_test']:7.4f} {r['auc_image']:7.4f} {r['bacc_image']:7.4f} "
              f"{r['auc_case']:7.4f} {r['bacc_case']:7.4f} {r['gap']:+8.4f}")

    # ---- E3: is the S5 KG-Laplacian effect real at this sample size? -------
    print("\n=== E3: S5 confirmatory ===")
    A, _ = build_adjacency(_KG, names)
    same = (prior[:, None] * prior[None, :]) > 0
    As = A * same - A * (~same)
    Lap = np.diag(np.abs(As).sum(1)) - As
    # refit calibration on BreastMNIST train under each graph, apply here unchanged
    sys.path.insert(0, str(_HERE.parent))
    from freeze_models import breastmnist_matrix                       # noqa: E402
    Xb, Yb = breastmnist_matrix([(n, p) for n, p in zip(names, prior)])
    Zb = {s: (Xb[s] - mu) / sd for s in Xb}
    nb, Fn = len(TAGS), len(names)
    l2 = F["models"]["s5"]["l2"]

    def fit_apply(L, lam):
        a, c, w, b, ix = fit_calib(Zb["train"], Yb["train"], Fn, nb, L, lam, l2)
        ix2 = np.tile(np.arange(Fn), nb)
        return (Z * a[ix2] + c[ix2]) @ w + b

    s_kg = fit_apply(Lap, F["models"]["s5"]["lambda"])
    s_l0 = fit_apply(np.zeros_like(Lap), 0.0)
    sh = []
    for k in range(5):
        rng = np.random.default_rng(200 + k)
        iu = np.triu_indices(Fn, 1); w_ = As[iu].copy(); rng.shuffle(w_)
        B = np.zeros_like(As); B[iu] = w_; B = B + B.T
        sh.append(fit_apply(np.diag(np.abs(B).sum(1)) - B, F["models"]["s5"]["lambda"]))
    u = np.unique(cases)

    def cs(v):
        return np.array([v[cases == cc].mean() for cc in u])
    y_case = np.array([y[cases == cc][0] for cc in u])
    a_kg, a_l0 = auc(y_case, cs(s_kg)), auc(y_case, cs(s_l0))
    a_sh = [auc(y_case, cs(v)) for v in sh]
    p_l0 = paired_bootstrap(y_case, cs(s_kg), cs(s_l0))
    p_sh = paired_bootstrap(y_case, cs(s_kg), cs(np.mean(sh, 0)))
    print(f"  KG-signed Laplacian  case AUC {a_kg:.4f}")
    print(f"  lambda = 0           case AUC {a_l0:.4f}   delta {a_kg-a_l0:+.4f}"
          f"   P(better) {p_l0:.3f}")
    print(f"  shuffled KG (5)      case AUC {np.mean(a_sh):.4f} +- {np.std(a_sh):.4f}"
          f"   delta {a_kg-np.mean(a_sh):+.4f}   P(better) {p_sh:.3f}")
    ok = p_l0 >= 0.95 and p_sh >= 0.95
    print(f"  -> registered bar P>=0.95: {'CONFIRMED' if ok else 'NOT confirmed'}")
    res["E3"] = {"auc_kg": a_kg, "auc_lambda0": a_l0, "auc_shuffled_mean": float(np.mean(a_sh)),
                 "auc_shuffled_sd": float(np.std(a_sh)), "P_vs_lambda0": p_l0,
                 "P_vs_shuffled": p_sh, "confirmed": bool(ok)}

    # ---- E4: agreement with radiologist BI-RADS ---------------------------
    print("\n=== E4: agreement with radiologist BI-RADS ===")
    res["E4"] = {}
    for key in ("kgsum", "logit", "s5"):
        s = scores[key][0]
        rho = spearman(s, birads.astype(float))
        hi = np.isin(birads, [4, 5])
        a45 = auc(hi.astype(int), s)
        print(f"  {key:8s} Spearman rho {rho:+.4f}   AUC(BI-RADS 4/5 vs 2/3) {a45:.4f}")
        res["E4"][key] = {"spearman": rho, "auc_birads45": a45}

    (_HERE / "results/e1_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote results/e1_analysis.json")


if __name__ == "__main__":
    main()
