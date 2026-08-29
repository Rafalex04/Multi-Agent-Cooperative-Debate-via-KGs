"""Replace the coupled r = 0.994 with an uncoupled statistic.

The reported Pearson r between in-domain AUC and the TRANSFER GAP is
mathematically coupled: gap = indomain - external, so in-domain appears on both
sides of the correlation. Oldham's problem. The coupling is made worse here
because the external range (0.0472) is far narrower than the in-domain range
(0.1278), so the gap is dominated by its in-domain term and the correlation is
close to 1 almost by construction.

The honest statistic is the correlation between the two MEASURED quantities,
in-domain and external, neither of which contains the other. With n = 6 the exact
permutation distribution is small enough to enumerate, so the p-value is exact
rather than asymptotic.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np

_H = Path(__file__).resolve().parent
B = json.loads((_H.parents[0] / "22_transfer/results/gap_law.json").read_text())["blocks"]


def rank(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r


def spearman(a, b):
    return float(np.corrcoef(rank(np.asarray(a, float)), rank(np.asarray(b, float)))[0, 1])


def main():
    ind = np.array([b["indomain"] for b in B])
    ext = np.array([b["external"] for b in B])
    gap = ind - ext
    n = len(ind)

    r_coupled = float(np.corrcoef(ind, gap)[0, 1])
    rho = spearman(ind, ext)
    # exact permutation p-value, two-sided, over all n! orderings
    perms = [spearman(ind, np.array(p)) for p in itertools.permutations(ext)]
    p_two = float(np.mean(np.abs(np.array(perms)) >= abs(rho)))

    print(f"n = {n} feature blocks (constructed, not sampled)")
    print(f"  in-domain range {ind.max()-ind.min():.4f}   external range {ext.max()-ext.min():.4f}"
          f"   ratio {(ind.max()-ind.min())/(ext.max()-ext.min()):.2f}x")
    print(f"\n  COUPLED   Pearson(in-domain, gap)      {r_coupled:+.3f}   <- do not use")
    print(f"            gap = in-domain - external, so in-domain is on both sides")
    print(f"\n  UNCOUPLED Spearman(in-domain, external) {rho:+.3f}")
    print(f"            exact two-sided permutation p = {p_two:.4f} over {len(perms)} orderings")
    print(f"            (the strongest possible |rho| at n=6 is 1.000, p = {2/len(perms):.4f})")
    print(f"\n  Pearson(in-domain, external)            {float(np.corrcoef(ind, ext)[0,1]):+.3f}")
    print(f"  Spearman(params, gap)                   {spearman([b['params'] for b in B], gap):+.3f}")
    (_H / "results/gap_correction.json").write_text(json.dumps(
        {"n": n, "pearson_indomain_gap_COUPLED": r_coupled,
         "spearman_indomain_external": rho, "perm_p_two_sided": p_two,
         "n_permutations": len(perms),
         "pearson_indomain_external": float(np.corrcoef(ind, ext)[0, 1]),
         "indomain_range": float(ind.max()-ind.min()),
         "external_range": float(ext.max()-ext.min()),
         "spearman_params_gap": spearman([b["params"] for b in B], gap)},
        indent=1))
    print("\nwrote results/gap_correction.json")


if __name__ == "__main__":
    main()
