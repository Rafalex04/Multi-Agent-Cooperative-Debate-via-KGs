"""Vocabulary or semantics? The control that separates the project's two claims.

C3 says the KG's *edges* are a null. C4-adjacent reasoning says the KG pays
through its *vocabulary* -- the finding list that tells the probes what to ask.
Neither isolates the ontology's SEMANTICS: the statement that `arborizing_vessels`
means bcc, or that `venous_beading` means severe NPDR.

Three arms, identical in every other respect:

  real prior      the ontology's own finding -> class/level assignment
  shuffled prior  the SAME findings, the SAME probes, the SAME graph, with the
                  prior vector PERMUTED across findings. Vocabulary intact,
                  semantics destroyed. 5 draws.
  no prior        the prior channel and penalty centre zeroed entirely.

If shuffled matches real, the ontology contributes a vocabulary and nothing more,
and the thesis should say so plainly. If real beats shuffled, the semantics carry
signal that the probe list alone does not -- which would be the first positive
KG result in the project and needs to be reported as carefully as the nulls.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "shared"))
from transfer_matrix import (load_breast, load_retina, load_derma,     # noqa: E402
                             score, TRUNK)
sys.path.insert(0, str(REPO / "retinaMnist/src/retina_kg"))
sys.path.insert(0, str(REPO / "dermaMnist/src/derma_kg"))
import ordinal, nominal                                              # noqa: E402


def fit_full(D, prior, l2, k, seed, epochs):
    if D["kind"] == "ordinal":
        return ordinal.fit(D["Z"]["train"], D["Y"]["train"], D["Ap"], D["An"],
                           D["n_cut"], l2=l2, kdim=k, epochs=epochs, seed=seed,
                           prior_centre=prior)
    return nominal.fit(D["Z"]["train"], D["Y"]["train"], D["Ap"], D["An"],
                       D["n_class"], l2=l2, kdim=k, epochs=epochs, seed=seed,
                       prior_centre=prior, class_weight="balanced")


def with_prior(D, prior):
    """Rebuild the tensors so the PRIOR CHANNEL matches the prior being tested.

    The prior appears twice -- as a node feature and as the readout penalty
    centre -- so shuffling only the centre would leave the feature channel
    telling the truth and the control would be toothless.
    """
    E = dict(D)
    Z = {s: D["Z"][s].copy() for s in D["Z"]}
    col = -1 if D["kind"] == "ordinal" else 3
    if prior is None:
        for s in Z:
            Z[s][:, :, col] = 0.0
    elif D["kind"] == "ordinal":
        for s in Z:
            Z[s][:, :, col] = np.asarray(prior)[None, :]
    else:
        spec = 1.0 / np.maximum(1e-9, (np.asarray(prior) > 0).sum(1))
        for s in Z:
            Z[s][:, :, col] = spec[None, :]
    E["Z"] = Z
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="breast,retina,derma")
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--kdim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "shared/results/prior_shuffle.json"))
    a = ap.parse_args()

    LOAD = {"breast": load_breast, "retina": load_retina, "derma": load_derma}
    res = {"config": {"l2": a.l2, "kdim": a.kdim, "epochs": a.epochs,
                      "draws": a.draws}}
    print(f"{'dataset':8s} {'metric':8s} {'real':>9s} {'shuffled prior':>20s} "
          f"{'no prior':>9s} {'real-shuf':>11s}")
    for name in a.datasets.split(","):
        D = LOAD[name]()
        pr = np.asarray(D["prior"], float)
        real = score(with_prior(D, pr),
                     fit_full(with_prior(D, pr), pr, a.l2, a.kdim, a.seed, a.epochs))
        rng = np.random.default_rng(a.seed)
        sh = []
        for _ in range(a.draws):
            perm = rng.permutation(len(pr))
            ps = pr[perm]
            sh.append(score(with_prior(D, ps),
                            fit_full(with_prior(D, ps), ps, a.l2, a.kdim,
                                     a.seed, a.epochs)))
        zero = np.zeros_like(pr)
        nop = score(with_prior(D, None),
                    fit_full(with_prior(D, None), zero, a.l2, a.kdim, a.seed, a.epochs))
        m, s = float(np.mean(sh)), float(np.std(sh))
        z = (real - m) / s if s > 0 else float("nan")
        res[name] = {"metric": D["metric"], "real": real,
                     "shuffled": {"mean": m, "sd": s, "draws": sh},
                     "no_prior": nop, "delta": real - m, "z": z}
        print(f"{name:8s} {D['metric']:8s} {real:9.4f} {m:12.4f} +-{s:6.4f} "
              f"{nop:9.4f} {real-m:+8.4f} ({z:+.2f} sd)")
        del D
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
