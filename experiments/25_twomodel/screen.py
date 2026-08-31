"""Partner selection, exactly as pre-registered on 2026-08-30.

Zero-parameter KG-signed sum on a BreastMNIST TRAIN subsample (n=250).
The test set and BUS-BRA are not touched here. Nothing is fitted.
Bar: AUC >= 0.60. Below that a model is noise injection, not a debater.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
for _d in ("16_external", "14_kgtensor"):
    sys.path.insert(0, str(_HERE.parents[0] / _d))
from analyse_e1 import auc, bacc, spearman          # noqa: E402
from claims import kg_findings                      # noqa: E402

BAR = 0.60
INCUMBENT = "qwen3vl"
CANDIDATES = ["gemma3", "medgemma", "minicpm", "llama32v"]
TAGS = {"qwen3vl": "qwen3-vl:8b-instruct", "gemma3": "gemma3:12b",
        "medgemma": "medgemma:4b", "minicpm": "minicpm-v:8b",
        "llama32v": "llama3.2-vision:11b"}


def load(tag, names):
    out = {}
    for f in sorted((_HERE / "results").glob(f"screen_{tag}_*.jsonl")):
        for ln in f.read_text(errors="replace").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            v = r.get("p_yes") or {}
            if not any(v.get(n) is not None for n in names):
                continue
            out[int(r["index"])] = np.array(
                [float(v[n]) if v.get(n) is not None else 0.5 for n in names])
    return out


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)

    z = np.load(_HERE.parents[1] / "data/external/screen_train250_224.npz")
    y_all = (z["labels"][:, 0] == 0).astype(float)          # MALIGNANT=0 -> y=1

    P = {t: load(t, names) for t in [INCUMBENT] + CANDIDATES}
    P = {t: v for t, v in P.items() if v}
    common = sorted(set.intersection(*[set(v) for v in P.values()]))
    print(f"models with data: {list(P)}")
    print(f"common images: {len(common)}  malignant {int(y_all[common].sum())}\n")
    if len(common) < 50:
        print("too few common images to select on; wait for the screen"); return

    y = y_all[common]
    res, S = {}, {}
    for t, d in P.items():
        M = np.array([d[i] for i in common])
        Z = (M - M.mean(0)) / np.where(M.std(0) < 1e-9, 1, M.std(0))
        s = Z @ prior
        S[t] = s
        res[t] = {"model": TAGS[t], "n": len(common), "auc": float(auc(y, s)),
                  "bacc": float(bacc(y, s, float(np.median(s)))),
                  "mean_p": float(M.mean()), "sd_p": float(M.std(0).mean())}

    print(f"{'model':24s} {'AUC':>7s} {'bAcc':>7s} {'mean p':>8s} {'sd':>7s}  bar>=0.60")
    order = sorted(res, key=lambda t: -res[t]["auc"])
    for t in order:
        r = res[t]
        mark = "INCUMBENT" if t == INCUMBENT else ("PASS" if r["auc"] >= BAR else "FAIL")
        print(f"{r['model']:24s} {r['auc']:7.4f} {r['bacc']:7.4f} "
              f"{r['mean_p']:8.4f} {r['sd_p']:7.4f}  {mark}")

    # decorrelation from the incumbent - the mechanism under test
    print(f"\nper-finding probe correlation with {TAGS[INCUMBENT]} (Spearman, mean over findings)")
    corr = {}
    for t in order:
        if t == INCUMBENT:
            continue
        A, B = np.array([P[t][i] for i in common]), np.array([P[INCUMBENT][i] for i in common])
        cs = [spearman(A[:, k], B[:, k]) for k in range(len(names))]
        corr[t] = float(np.mean(cs))
        res[t]["corr_incumbent"] = corr[t]
        print(f"  {TAGS[t]:24s} {corr[t]:+.4f}")

    passing = [t for t in order if t != INCUMBENT and res[t]["auc"] >= BAR]
    print("\nDECISION")
    if not passing:
        print(f"  no candidate clears AUC {BAR:.2f} -> H1 UNTESTABLE on available models")
        pick = None
    else:
        best = max(res[t]["auc"] for t in passing)
        tied = [t for t in passing if best - res[t]["auc"] < 0.01]
        pick = min(tied, key=lambda t: corr[t]) if len(tied) > 1 else tied[0]
        print(f"  clears the bar: {', '.join(TAGS[t] for t in passing)}")
        if len(tied) > 1:
            print(f"  tie within 0.01 -> broken on lower correlation with the incumbent")
        print(f"  PARTNER = {TAGS[pick]}  (AUC {res[pick]['auc']:.4f}, "
              f"corr {corr[pick]:+.4f})")
    (_HERE / "results" / "screen.json").write_text(json.dumps(
        {"bar": BAR, "n": len(common), "per_model": res, "partner": pick and TAGS[pick],
         "partner_tag": pick}, indent=1))


if __name__ == "__main__":
    main()
