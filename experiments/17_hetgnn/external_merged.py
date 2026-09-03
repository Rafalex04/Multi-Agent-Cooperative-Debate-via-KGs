"""The full merged architecture, tested outside its training distribution.

Everything the brief asked for, in one object and on data it has never seen:

    claim --addresses(AGREE:+1 / DISAGREE:-1)--> claim     the debate graph
    claim --cites-->                             finding    the merge
    finding --kg(signed by stance agreement)-->  finding    the knowledge graph
    GNN message passing over the finding nodes, fed by both

Until now this had only ever been evaluated on 156 BreastMNIST test samples,
where every effect in this project has been decided at 1-2 sd. Here it runs on
BUS-BRA -- different country, four different scanners -- with debates generated
by the identical runner and KG, and with folds split on CASE so the two views of
a lesion cannot straddle a boundary.

The arms are the ones that decide something rather than the ones that flatter:

    probes only        no debate channel at all
    probes + debate    the merge, flat readout
    GNN + real KG      message passing on the ontology
    GNN + shuffled KG  same density, rewired -- isolates ontology from graph-ness
    GNN + no graph     isolates message passing from capacity

PERCEPTION REPAIR (--drop-inverted). Five of the sixteen probes anti-correlate
with the radiologist annotation of the descriptor they name, measured on BrEaST
(experiments/20_perception). Those five are then multiplied by a fixed KG stance,
so they are subtracted where they should be added. `--drop-inverted` weights them
zero -- in the probe channels, in the prior channel and in the adjacency signs --
which is the conservative repair: it claims the questions are broken, not that
they are backwards. The five are chosen on a DIFFERENT dataset against a
DIFFERENT label (descriptor agreement, not malignancy), so nothing about BUS-BRA
enters the choice.
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "15_kggnn"))
sys.path.insert(0, str(_HERE.parents[1] / "16_external"))
from analyse_e1 import auc, load_probes, paired_bootstrap                # noqa: E402
from claims import kg_findings                                           # noqa: E402
from gates import fit_rank                                               # noqa: E402
from hetgraph import CMAX, kg_operators, load_sample                     # noqa: E402
from powered_gnn import case_folds                                       # noqa: E402
from run_thothgnn3 import shuffle_kg                                     # noqa: E402
from sparsity import topk                                                # noqa: E402
from thothgnn3 import fit, forward                                       # noqa: E402

_DEB = _HERE.parents[2] / "breastMnist/data/breast/debates_busbra/all"
_EXT = _HERE.parents[1] / "16_external/results"
_MASKRES = _HERE.parents[1] / "23_masktest/results"

# The mask arm differs from auto in the IMAGE only: same backbone, same rounds,
# same protocol. Corpus, probe tags and npz must switch together or the gold
# cross-check below fires.
ARMS = {"auto": ("debates_busbra", ("busbra_p2", "busbra_p3"), _EXT,
                 "busbra_pad2_224.npz"),
        "mask": ("debates_busbra_mask", ("bring_p2", "bring_p3"), _MASKRES,
                 "busbra_bring_224.npz")}


def load_debates(names, deb_dir=None):
    """index -> (claim features, signed claim->claim, claim->finding, mask)."""
    out = {}
    for f in sorted(glob.glob(str((deb_dir or _DEB) / "*.json"))):
        try:
            X, Acc, Acf, mask, y, sid = load_sample(f, names)
        except Exception:
            continue
        out[int(sid)] = (X, Acc, Acf, mask, y)
    return out


INVERTED = ("irregular_shape", "echogenic_pseudocapsule", "oval_shape",
            "thin_uniform_pseudocapsule", "echogenic_rind")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-inverted", action="store_true")
    ap.add_argument("--contestation", action="store_true",
                    help="add the per-finding disagreement channel")
    ap.add_argument("--phrasings", default="both", choices=("both", "p2", "p3"),
                    help="which probe phrasing arms feed the finding nodes. "
                         "The negated arm P2 scores 0.4882 against the radiologist "
                         "descriptors on BrEaST -- chance -- and 0.4452 on BUS-BRA "
                         "malignancy through the KG rule, so `both` is averaging a "
                         "good channel against an anti-predictive one.")
    ap.add_argument("--arm", default="auto", choices=tuple(ARMS))
    ap.add_argument("--out", default="external_merged.json")
    args = ap.parse_args()

    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)
    keep = np.ones(len(names))
    if args.drop_inverted:
        for f in INVERTED:
            keep[names.index(f)] = 0.0
        prior = prior * keep
        print(f"perception repair ON: dropping {int((keep==0).sum())} inverted probes "
              f"-> {list(INVERTED)}")

    _dname, _tags, _pdir, _npz = ARMS[args.arm]
    _debdir = _HERE.parents[2] / "breastMnist/data/breast" / _dname / "all"
    print(f"arm={args.arm}  debates={_dname}  probes={_tags}  npz={_npz}")
    deb = load_debates(names, _debdir)
    probes = load_probes(names, _tags, _pdir)
    common = sorted(set(deb) & set(probes))
    print(f"BUS-BRA debates {len(deb)}  probes {len(probes)}  BOTH {len(common)}")
    if len(common) < 200:
        print("not enough debates yet"); return

    z = np.load(_HERE.parents[2] / "data/external" / _npz)
    y = (z["labels"][:, 0] == 0).astype(float)[common]
    cases = z["cases"][common]
    # cross-check the debate's own gold against the npz, since a silent mismatch
    # here would invalidate everything downstream
    mism = sum(1 for i in common if deb[i][4] != y[list(common).index(i)]) if False else \
        sum(1 for k, i in enumerate(common) if deb[i][4] != y[k])
    print(f"debate gold vs npz label mismatches: {mism}")
    assert mism == 0, "label misalignment between debate corpus and npz"

    F = len(names)
    n = len(common)
    # channels: probe P2, probe P3, argument mass, net stance, prior [, contestation]
    #
    # Net stance conflates "three claims, all malignant" with "five malignant and
    # two benign": both sum to +3. Whether the two agents DISAGREED about finding
    # f is a per-finding quantity that no probe can express and that the flat
    # vector currently throws away. Contestation is its binary entropy, which is
    # 0 when a finding is uncontested however much mass it carries, and maximal
    # when the mass splits evenly.
    npr = 2 if args.phrasings == "both" else 1
    nch = (5 if npr == 2 else 4) + (1 if args.contestation else 0)
    PMASS, PSTANCE, PPRIOR = npr, npr + 1, npr + 2
    PCONT = npr + 3
    X = np.zeros((n, F, nch))
    for k, i in enumerate(common):
        Xc, Acc, Acf, mask, _ = deb[i]
        pr = probes[i] if args.phrasings == "both" else \
            probes[i][:, 0:1] if args.phrasings == "p2" else probes[i][:, 1:2]
        X[k, :, 0:npr] = pr * keep[:, None]
        X[k, :, PMASS] = (mask[:, None] * Acf).sum(0)
        X[k, :, PSTANCE] = ((mask * Xc[:, 0])[:, None] * Acf).sum(0)
        X[k, :, PPRIOR] = prior
        if args.contestation:
            npos = ((mask * (Xc[:, 0] > 0))[:, None] * Acf).sum(0)
            nneg = ((mask * (Xc[:, 0] < 0))[:, None] * Acf).sum(0)
            tot = npos + nneg
            q = np.divide(npos, tot, out=np.zeros(F), where=tot > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
            X[k, :, PCONT] = np.where((tot > 0) & (q > 0) & (q < 1), h, 0.0)
    mu = X.reshape(-1, nch).mean(0); sd = X.reshape(-1, nch).std(0)
    Z = (X - mu) / np.where(sd < 1e-9, 1, sd)

    Ap, An = kg_operators(names, prior)
    Zero = np.zeros_like(Ap)
    folds = case_folds(cases, k=5, seed=0)
    u = np.unique(cases)
    y_case = np.array([y[cases == c][0] for c in u])
    def to_case(v):
        return np.array([v[cases == c].mean() for c in u])

    shuf = [shuffle_kg(Ap, An, np.random.default_rng(700 + s)) for s in range(5)]
    arms = {"GNN+KG": (Ap, An), "GNN+nograph": (Zero, Zero)}
    for s in range(5):
        arms[f"shuf{s}"] = shuf[s]
    labs = ["probes only", "probes + debate"]
    if args.contestation:
        labs += ["probes + contestation", "probes + debate + cont"]
    oof = {k: np.zeros(n) for k in list(arms) + labs}

    for te in folds:
        tr = ~te
        # flat arms, with and without the debate columns
        pc = list(range(npr))
        flat = [("probes only", pc + [PPRIOR]),
                ("probes + debate", pc + [PMASS, PSTANCE, PPRIOR])]
        if args.contestation:
            flat.append(("probes + contestation", pc + [PPRIOR, PCONT]))
            flat.append(("probes + debate + cont", pc + [PMASS, PSTANCE, PPRIOR, PCONT]))
        for lab, cols in flat:
            A = Z[:, :, cols].reshape(n, -1)
            w, b = fit_rank(A[tr], y[tr], 0.3)
            oof[lab][te] = A[te] @ w + b
        for lab, g in arms.items():
            P = fit(Z[tr], y[tr], g[0], g[1], l2=0.03, kdim=2, prior_centre=prior)
            oof[lab][te] = forward(P, Z[te], g[0], g[1])[2]

    print(f"\nn={n} images, {len(u)} cases, malignant {y.mean():.3f}")
    print(f"{'arm':22s} {'image AUC':>10s} {'case AUC':>10s}")
    res = {"n": n, "n_cases": int(len(u))}
    for lab in labs + ["GNN+nograph", "GNN+KG"]:
        ai, ac = auc(y, oof[lab]), auc(y_case, to_case(oof[lab]))
        res[lab] = {"image": ai, "case": ac}
        print(f"{lab:22s} {ai:10.4f} {ac:10.4f}")
    sh_c = [auc(y_case, to_case(oof[f"shuf{s}"])) for s in range(5)]
    print(f"{'GNN+shuffled KG':22s} {'':>10s} {np.mean(sh_c):10.4f}   "
          f"(sd {np.std(sh_c):.4f}, 5 draws)")

    d_deb = res["probes + debate"]["case"] - res["probes only"]["case"]
    p_deb = paired_bootstrap(y_case, to_case(oof["probes + debate"]),
                             to_case(oof["probes only"]))
    d_kg = res["GNN+KG"]["case"] - float(np.mean(sh_c))
    p_kg = paired_bootstrap(y_case, to_case(oof["GNN+KG"]),
                            to_case(np.mean([oof[f"shuf{s}"] for s in range(5)], 0)))
    print(f"\n  debate on top of probes : {d_deb:+.4f}   P(better) {p_deb:.3f}")
    print(f"  real KG vs shuffled KG  : {d_kg:+.4f}   "
          f"({d_kg/(np.std(sh_c)+1e-9):+.2f} sd)   P(better) {p_kg:.3f}")
    res.update({"shuffled_case_mean": float(np.mean(sh_c)),
                "shuffled_case_sd": float(np.std(sh_c)),
                "debate_delta": d_deb, "P_debate": p_deb,
                "kg_delta": d_kg, "P_kg": p_kg})
    print("\n  -> debate: " + ("HELPS" if p_deb >= 0.95 else "no effect at P>=0.95"))
    print("  -> KG topology: " + ("HELPS" if p_kg >= 0.95 else "no effect at P>=0.95"))
    # case-level score vectors, so the cross-method paired bootstraps can be run
    # against the baselines without refitting anything.
    np.savez(str((_HERE.parent / "results" / args.out).with_suffix(".scores.npz")),
             y_case=y_case, cases=u,
             **{lab.replace(" ", "_"): to_case(oof[lab])
                for lab in labs + ["GNN+nograph", "GNN+KG"]})
    (_HERE.parent / "results" / args.out).write_text(
        json.dumps(res, indent=1, default=float))
    print(f"\nwrote results/{args.out}")


if __name__ == "__main__":
    main()
