"""Finding-to-finding adjacency over the dermoscopy graph.

Identical Adamic-Adar formula to breast and retina -- two findings are adjacent
when they share a mediator, weighted so a mediator linking few findings counts
for more:

    A[i,j] = sum over shared mediators M of  1 / log(1 + |findings(M)|)

Only what counts as a mediator changes, and it comes from the yaml.

THE SIGNED SPLIT IS SET-BASED, NOT SIGN-BASED. Breast split A by the product of
two +-1 stance priors; retina by ordinal level proximity. With seven unordered
classes neither applies, so:

    A+   the two findings' class sets OVERLAP   (they can indicate the same thing)
    A-   the two findings' class sets are DISJOINT

Unlike ICDR -- which had 18 appearance triples, 5 of them placeholders, so the
breast-style appearance construction collapsed to 1 edge -- dermoscopy has 53
real `dermoscopic_appearance_includes` triples, so that arm is expected to work
here. Both are built and compared.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_STOP = {"the", "and", "or", "of", "in", "to", "at", "a", "is", "with", "no",
         "stated", "morphology", "lesion", "structures", "structure", "area",
         "areas", "pattern"}


def _tokens(s):
    return {t for t in str(s).lower().replace("-", "_").split("_")
            if t and t not in _STOP and len(t) > 3}


def mediator_sets(triples, findings, cfg, groups=None):
    groups = list(groups if groups is not None else cfg["adjacency_mediators"])
    rel_of = cfg["adjacency_relations"]
    names = [f[0] for f in findings]
    idx = {n: i for i, n in enumerate(names)}
    wanted = {r for g in groups for r in rel_of.get(g, [])}
    med = defaultdict(set)
    for t in triples:
        rel, s, o = str(t["relation"]), str(t["subject"]), str(t["object"])
        if rel not in wanted:
            continue
        if s in idx:
            med[idx[s]].add(f"{rel}:{o}")
        if o in idx:                      # symmetric: a shared partner joins both
            med[idx[o]].add(f"{rel}:{s}")
    return med, names


def appearance_sets(triples, findings, cfg):
    """The breast-identical construction: mediator = a shared descriptor token."""
    names = [f[0] for f in findings]
    placeholder = cfg["appearance_placeholder"]
    desc = defaultdict(set)
    for t in triples:
        if str(t["relation"]) in cfg["appearance_relations"]:
            if str(t["object"]) == placeholder:
                continue
            desc[str(t["subject"])] |= _tokens(t["object"])
    med = defaultdict(set)
    for lesion, toks in desc.items():
        for i, n in enumerate(names):
            if _tokens(n) & toks or n == lesion:
                med[i] |= {f"appearance:{t}" for t in toks}
    return med, names


def build_adjacency(pack_dir, cfg, mediators=None, normalise=True):
    pack = Path(pack_dir)
    triples = json.loads((pack / "knowledge_graph.json").read_text())["triples"]
    fd = json.loads((pack / "findings.json").read_text())["findings"]
    findings = [(f["feature"], f["classes"], f["prior"]) for f in fd]

    if mediators == ["appearance"]:
        med, names = appearance_sets(triples, findings, cfg)
    else:
        med, names = mediator_sets(triples, findings, cfg, mediators)

    deg = defaultdict(int)
    for i in med:
        for m in med[i]:
            deg[m] += 1

    F = len(names)
    A = np.zeros((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            w = sum(1.0 / np.log1p(1 + deg[m]) for m in (med[i] & med[j]))
            A[i, j] = A[j, i] = w
    if normalise and A.max() > 0:
        A /= A.max()
    info = {"n_findings": F, "n_edges": int((A > 0).sum() // 2),
            "n_pairs": F * (F - 1) // 2, "n_mediators": len(deg),
            "isolated": [names[i] for i in range(F) if A[i].sum() == 0],
            "mediators": mediators or cfg["adjacency_mediators"]}
    return A, info


def kg_operators(A, classes):
    """A+ where class sets overlap, A- where they are disjoint."""
    sets = [set(c) for c in classes]
    F = len(sets)
    same = np.zeros((F, F), bool)
    for i in range(F):
        for j in range(F):
            same[i, j] = bool(sets[i] & sets[j])
    return A * same, A * (~same)


if __name__ == "__main__":
    import argparse, sys, yaml
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(P.PACK))
    ap.add_argument("--ontology", default=str(P.ONTOLOGY))
    ap.add_argument("--mediators", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.ontology).read_text())
    fd = json.loads((Path(a.pack) / "findings.json").read_text())["findings"]
    classes = [f["classes"] for f in fd]; names = [f["feature"] for f in fd]
    for meds in ([None, ["appearance"]] if not a.mediators
                 else [a.mediators.split(",")]):
        A, info = build_adjacency(a.pack, cfg, meds)
        Ap, An = kg_operators(A, classes)
        lab = "appearance" if meds == ["appearance"] else "structural"
        print(f"\n{lab}: {info['n_edges']}/{info['n_pairs']} pairs, "
              f"{len(info['isolated'])} isolated, {info['n_mediators']} mediators")
        print(f"  A+ {int((Ap>0).sum()//2)} edges   A- {int((An>0).sum()//2)} edges")
        iu = np.triu_indices(len(names), 1)
        for k in np.argsort(-A[iu])[:5]:
            i, j = iu[0][k], iu[1][k]
            if A[i, j] <= 0:
                break
            print(f"    {A[i,j]:.3f}  {names[i]:32s} -- {names[j]}")
