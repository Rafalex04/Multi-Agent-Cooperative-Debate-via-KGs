"""Finding-to-finding adjacency, derived from the knowledge graph.

Three attempts, and the first two failed for reasons worth recording.

DIRECT TRIPLES. There are none. The 16 debate findings have zero triples
between them -- in this graph they are leaf terms.

1-HOP NEIGHBOURS. Connects 94 of 120 pairs, but only through
`suggests_malignancy true` and `suggests_benign true`. That is stance, not
clinical structure, and message passing over it is pure label homophily. A
degree cutoff to remove those hubs is self-defeating: the lesion entities that
genuinely link findings are themselves high-degree, so cutting hubs left 1 edge.

LESION-MEDIATED. What the graph actually knows about co-occurrence is in its
lesion descriptions -- 113 `ultrasound_appearance_includes` triples and 8
`combination_indicates`. `well_circumscribed_oval_hypoechoic_uniform` describes
fibroadenoma and names both `circumscribed` and `oval`; a simple cyst is
`anechoic_with_posterior_enhancement_thin_wall_circumscribed`. So two findings
are adjacent when some lesion presents both, weighted so that a lesion
presenting few findings counts for more than one presenting many:

    A[i,j] = sum over shared lesions L of  1 / log(1 + |findings(L)|)

That connects 72 of 120 pairs and the strongest links are the clinically
correct ones (echogenic pseudocapsule with echogenic rind; circumscribed with
oval; anechoic with thin uniform capsule; posterior shadowing with
microcalcifications).

Nothing here names a finding or a lesion, so pointing it at another ontology
with appearance triples yields that ontology's adjacency.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_APPEARANCE = "ultrasound_appearance_includes"
_COMBINATION = "combination_indicates"
_STOP = {"mass", "with", "or", "and", "of", "the", "under", "over", "no",
         "in", "to", "at", "a", "is", "type", "breast"}


def _tokens(s):
    return {t for t in str(s).lower().split("_") if t and t not in _STOP and len(t) > 3}


def lesion_descriptions(triples):
    """lesion -> the set of descriptor tokens the graph attributes to it."""
    desc = defaultdict(set)
    for t in triples:
        r = str(t.get("relation"))
        if r == _APPEARANCE:
            desc[str(t["subject"])] |= _tokens(t["object"])
        elif r == _COMBINATION:
            # combination_indicates points the other way: descriptor -> lesion
            desc[str(t["object"])] |= _tokens(t["subject"])
    return desc


def build_adjacency(kg_path, findings, normalise=True):
    """(A, info). A is symmetric, zero diagonal, scaled to max 1."""
    triples = json.loads(Path(kg_path).read_text())["triples"]
    desc = lesion_descriptions(triples)

    f2l = defaultdict(set)
    for i, f in enumerate(findings):
        tf = _tokens(f)
        for les, dt in desc.items():
            if tf & dt:
                f2l[i].add(les)
    # how many findings each lesion presents -- the Adamic-Adar denominator
    l_deg = defaultdict(int)
    for i in f2l:
        for les in f2l[i]:
            l_deg[les] += 1

    F = len(findings)
    A = np.zeros((F, F))
    for i in range(F):
        for j in range(i + 1, F):
            shared = f2l[i] & f2l[j]
            w = sum(1.0 / np.log1p(1 + l_deg[L]) for L in shared)
            A[i, j] = A[j, i] = w
    if normalise and A.max() > 0:
        A /= A.max()
    info = {"n_edges": int((A > 0).sum() // 2),
            "n_pairs": F * (F - 1) // 2,
            "isolated": [findings[i] for i in range(F) if A[i].sum() == 0],
            "n_lesions": len(desc)}
    return A, info


def normalised(A, self_loops=True):
    """Symmetric normalisation D^-1/2 (A + I) D^-1/2, as in a GCN layer."""
    M = A + np.eye(len(A)) if self_loops else A.copy()
    d = M.sum(1)
    d = np.where(d <= 0, 1.0, d)
    Dm = np.diag(1.0 / np.sqrt(d))
    return Dm @ M @ Dm


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "14_kgtensor"))
    from claims import kg_findings
    F = [f for f, _ in kg_findings()]
    root = Path(__file__).resolve().parents[2] / "breastMnist/data/breast/knowledge_graph.json"
    A, info = build_adjacency(root, F)
    print(f"edges {info['n_edges']}/{info['n_pairs']}   lesions {info['n_lesions']}")
    print(f"isolated findings: {info['isolated']}")
    print("\nstrongest links:")
    seen = set()
    for i, j in np.dstack(np.unravel_index(np.argsort(-A, axis=None), A.shape))[0]:
        if A[i, j] == 0 or (j, i) in seen:
            continue
        seen.add((i, j))
        print(f"   {A[i,j]:.3f}  {F[i]}  <->  {F[j]}")
        if len(seen) >= 10:
            break
