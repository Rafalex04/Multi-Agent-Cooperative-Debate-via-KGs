"""Finding-to-finding adjacency over the ICDR graph.

The breast construction (`experiments/15_kggnn/kg_graph.py`) is lesion-mediated:
two findings are adjacent when some lesion's `ultrasound_appearance_includes`
description names both, Adamic-Adar weighted so a lesion presenting few findings
counts for more than one presenting many. Its docstring claims the construction
transfers to "any ontology with appearance triples".

ICDR does not have enough of them. There are 18 `fundus_appearance_includes`
triples and 5 are the placeholder `no_stated_morphology`, so the direct port
gives a near-empty graph. It is kept as `--mediators appearance` because the
negative KG-topology result on breast was measured under THAT construction and
changing it silently would not be a replication.

The primary construction keeps the identical Adamic-Adar formula and swaps only
what counts as a mediator, which ICDR has in quantity:

    A[i,j] = sum over shared mediators M of  1 / log(1 + |findings(M)|)

    criterion    the quantitative rules the two findings both appear in
    taxonomy     a shared is_a / subtype_of / component_of parent
    location     a shared retinal layer or structure
    association  co-listed, confused with, or a differential for one another

Severity level is NOT a default mediator; see ontology_retina.yaml for why.
Nothing here names a finding or a relation -- both come from the yaml.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_STOP = {"the", "and", "or", "of", "in", "to", "at", "a", "is", "with", "no",
         "stated", "morphology", "retinal", "retina", "fundus"}


def _tokens(s):
    return {t for t in str(s).lower().split("_") if t and t not in _STOP and len(t) > 3}


def mediator_sets(triples, criteria, findings, cfg, groups=None):
    """finding index -> the set of mediators the ontology attributes to it."""
    groups = list(groups if groups is not None else cfg["adjacency_mediators"])
    rel_of = cfg["adjacency_relations"]
    names = [f[0] for f in findings]
    idx = {n: i for i, n in enumerate(names)}
    med = defaultdict(set)

    wanted = {r for g in groups for r in rel_of.get(g, [])}
    for t in triples:
        rel = str(t["relation"])
        s, o = str(t["subject"]), str(t["object"])
        if rel in wanted:
            # symmetric: a shared parent/location/partner joins both endpoints
            if s in idx:
                med[idx[s]].add(f"{rel}:{o}")
            if o in idx:
                med[idx[o]].add(f"{rel}:{s}")
        # criteria are referenced by a FIELD, not only by a relation
        if "criterion" in groups and t.get("requires_criterion") and s in idx:
            med[idx[s]].add(f"criterion:{t['requires_criterion']}")

    if "criterion" in groups:
        for c in criteria:
            les = c.get("lesion")
            if les in idx:
                med[idx[les]].add(f"criterion:{c['id']}")
            for sib in (c.get("sibling_criteria") or []):
                if les in idx:
                    med[idx[les]].add(f"criterion:{sib}")
    return med, names


def appearance_sets(triples, findings, cfg):
    """The breast-identical construction: mediator = a shared descriptor token."""
    names = [f[0] for f in findings]
    idx = {n: i for i, n in enumerate(names)}
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
                med[i] |= {f"appearance:{tok}" for tok in toks}
    return med, names


def build_adjacency(pack_dir, cfg, mediators=None, normalise=True):
    """(A, info). Symmetric, zero diagonal, scaled to max 1 -- as on breast."""
    pack = Path(pack_dir)
    triples = json.loads((pack / "knowledge_graph.json").read_text())["triples"]
    criteria = json.loads((pack / "criteria.json").read_text())["criteria"]
    fd = json.loads((pack / "findings.json").read_text())["findings"]
    findings = [(f["feature"], f["level"], f["prior"]) for f in fd]

    if mediators == ["appearance"]:
        med, names = appearance_sets(triples, findings, cfg)
    else:
        med, names = mediator_sets(triples, criteria, findings, cfg, mediators)

    deg = defaultdict(int)
    for i in med:
        for m in med[i]:
            deg[m] += 1

    F = len(names)
    A = np.zeros((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            shared = med[i] & med[j]
            w = sum(1.0 / np.log1p(1 + deg[m]) for m in shared)
            A[i, j] = A[j, i] = w
    if normalise and A.max() > 0:
        A /= A.max()

    info = {"n_findings": F,
            "n_edges": int((A > 0).sum() // 2),
            "n_pairs": F * (F - 1) // 2,
            "n_mediators": len(deg),
            "isolated": [names[i] for i in range(F) if A[i].sum() == 0],
            "mediators": mediators or cfg["adjacency_mediators"]}
    return A, info


def kg_operators(A, prior):
    """Signed split, identical rule to hetgraph.kg_operators.

    `prior` is the ICDR level recentred on the referable-DR boundary, so the sign
    means "pushes above / below grade 2" and no finding sits at zero.
    """
    prior = np.asarray(prior, dtype=float)
    same = (prior[:, None] * prior[None, :]) > 0
    return A * same, A * (~same)


if __name__ == "__main__":
    import argparse, yaml, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(P.PACK))
    ap.add_argument("--ontology", default=str(P.ONTOLOGY))
    ap.add_argument("--mediators", default=None,
                    help="comma list, or 'appearance' for the breast-identical arm")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.ontology).read_text())
    meds = a.mediators.split(",") if a.mediators else None
    A, info = build_adjacency(a.pack, cfg, meds)
    fd = json.loads((Path(a.pack) / "findings.json").read_text())["findings"]
    prior = np.array([f["prior"] for f in fd])
    Ap, An = kg_operators(A, prior)
    print(json.dumps(info, indent=1))
    print(f"A+ {int((Ap > 0).sum() // 2)} edges   A- {int((An > 0).sum() // 2)} edges")
    names = [f["feature"] for f in fd]
    iu = np.triu_indices(len(names), 1)
    order = np.argsort(-A[iu])
    print("\nstrongest links:")
    for k in order[:10]:
        i, j = iu[0][k], iu[1][k]
        if A[i, j] <= 0:
            break
        print(f"  {A[i,j]:.3f}  {names[i]:24s} -- {names[j]}")
