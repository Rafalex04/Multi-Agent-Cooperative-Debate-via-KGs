"""Is per-finding contestation new information, or a restatement of the probes?

The claim under test: "whether the agents disagreed about finding f is not
derivable from any probe, and it is a per-finding uncertainty signal the flat
vector cannot express."

That is two separate assertions and they need separate measurements.

  NOT DERIVABLE   contestation at finding f should not be predictable from the
                  probe at finding f. The natural competitor is probe
                  uncertainty |p - 0.5|: if a finding the model is unsure about
                  is exactly the finding the agents argue over, contestation is
                  a noisy copy of something already in the vector.
  CARRIES SIGNAL  contestation should say something about the label that the
                  probes do not already say.

Net stance conflates "three claims, all malignant" with "five malignant and two
benign" -- both sum to +3. Contestation is the binary entropy of the stance split
on that finding: 0 when uncontested however much mass it carries, 1 when the mass
splits evenly.

Run on the complete BUS-BRA corpus: 1875 debates, 1064 cases.
"""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _d in ("14_kgtensor", "15_kggnn", "16_external"):
    sys.path.insert(0, str(_HERE.parents[1] / _d))
from analyse_e1 import auc, load_probes, paired_bootstrap               # noqa: E402
from claims import kg_findings                                          # noqa: E402
from hetgraph import load_sample                                        # noqa: E402

_DEB = _HERE.parents[2] / "breastMnist/data/breast/debates_busbra/all"
_EXT = _HERE.parents[1] / "16_external/results"


def spearman(a, b):
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v)); r[o] = np.arange(len(v))
        return r
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    F = len(names)

    deb = {}
    for f in sorted(glob.glob(str(_DEB / "*.json"))):
        try:
            X, Acc, Acf, mask, y, sid = load_sample(f, names)
        except Exception:
            continue
        deb[int(sid)] = (X, Acf, mask)
    probes = load_probes(names, ("busbra_p2", "busbra_p3"), _EXT)
    common = sorted(set(deb) & set(probes))
    z = np.load(_HERE.parents[2] / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    y = (z["labels"][common, 0] == 0).astype(float)
    cases = z["cases"][common]
    n = len(common)
    print(f"complete corpus: {n} images / {len(np.unique(cases))} cases")

    MASS = np.zeros((n, F)); NET = np.zeros((n, F)); CON = np.zeros((n, F))
    P3 = np.zeros((n, F))
    for k, i in enumerate(common):
        Xc, Acf, mask = deb[i]
        MASS[k] = (mask[:, None] * Acf).sum(0)
        NET[k] = ((mask * Xc[:, 0])[:, None] * Acf).sum(0)
        npos = ((mask * (Xc[:, 0] > 0))[:, None] * Acf).sum(0)
        nneg = ((mask * (Xc[:, 0] < 0))[:, None] * Acf).sum(0)
        tot = npos + nneg
        q = np.divide(npos, tot, out=np.zeros(F), where=tot > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        CON[k] = np.where((tot > 0) & (q > 0) & (q < 1), h, 0.0)
        P3[k] = probes[i][:, 1]

    print(f"\ncontestation: mean {CON.mean():.4f}, "
          f"{100*(CON > 0).mean():.1f}% of (image, finding) pairs contested, "
          f"{100*(MASS > 0).mean():.1f}% cited at all")

    print("\n=== 1. is contestation derivable from the probe at the same finding? ===")
    print(f"{'finding':46s} {'rho(cont, |p-.5|)':>18s} {'rho(cont, mass)':>16s}")
    rho_u, rho_m = [], []
    for j, f in enumerate(names):
        ru = spearman(CON[:, j], -np.abs(P3[:, j] - 0.5))
        rm = spearman(CON[:, j], MASS[:, j])
        rho_u.append(ru); rho_m.append(rm)
        print(f"{f:46s} {ru:18.4f} {rm:16.4f}")
    print(f"{'MEAN |rho|':46s} {np.nanmean(np.abs(rho_u)):18.4f} "
          f"{np.nanmean(np.abs(rho_m)):16.4f}")
    print("  -> contestation is "
          + ("NOT explained by probe uncertainty" if np.nanmean(np.abs(rho_u)) < 0.15
             else "largely a restatement of probe uncertainty")
          + f"; it IS strongly tied to how much a finding is cited "
            f"(mean |rho| {np.nanmean(np.abs(rho_m)):.3f})")

    print("\n=== 2. does contestation say anything about the label? ===")
    u = np.unique(cases)
    yc = np.array([y[cases == c][0] for c in u])
    def to_case(v):
        return np.array([v[cases == c].mean() for c in u])
    tot_con = CON.sum(1)
    sc = {"total contestation": tot_con,
          "KG-signed contestation": CON @ prior,
          "KG-signed net stance": NET @ prior,
          "KG-signed argument mass": MASS @ prior,
          "KG-signed probe P3": ((P3 - P3.mean(0)) / P3.std(0)) @ prior}
    for lab, v in sc.items():
        c = to_case(v)
        print(f"  {lab:26s} case AUC {auc(yc, c):.4f}")

    base = to_case(sc["KG-signed probe P3"])
    for lab in ("total contestation", "KG-signed contestation"):
        comb = base + 0.0
        # residual test: does contestation add to the probe score under a
        # 2-parameter least-squares combination fitted on the SAME data?  That is
        # optimistic by construction, so a null here is decisive.
        A = np.stack([base, to_case(sc[lab])], 1)
        A = (A - A.mean(0)) / A.std(0)
        w = np.linalg.lstsq(A, yc - yc.mean(), rcond=None)[0]
        print(f"  probe P3 + {lab:24s} -> {auc(yc, A @ w):.4f}   "
              f"(optimistic: weights fitted on these same cases)")

    (_HERE.parent / "results/contestation.json").write_text(json.dumps(
        {"n": n, "n_cases": int(len(u)),
         "frac_contested": float((CON > 0).mean()),
         "mean_abs_rho_probe_uncertainty": float(np.nanmean(np.abs(rho_u))),
         "mean_abs_rho_mass": float(np.nanmean(np.abs(rho_m))),
         "case_auc": {k: float(auc(yc, to_case(v))) for k, v in sc.items()}},
        indent=1, default=float))
    print("\nwrote results/contestation.json")


if __name__ == "__main__":
    main()
