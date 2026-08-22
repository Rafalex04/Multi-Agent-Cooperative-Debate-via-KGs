"""The KG's lesion layer, extracted as a probeable differential diagnosis.

Every probe in this project so far asks about a FINDING -- "does the mass show
spiculated margins". That uses one slice of the ontology: the 16 findings carrying
a `suggests_malignancy` / `suggests_benign` stance. It leaves the largest part of
the graph untouched: 31 named lesions carrying 113 `ultrasound_appearance_includes`
edges and, for many of them, a `typically_classified_as` edge to a BI-RADS
category.

That second slice is a differential diagnosis, and it can be measured the same
way: ask the model whether the image shows a simple cyst, a fibroadenoma, an oil
cyst, and read the first-token distribution. The ontology then converts each
lesion belief into a malignancy prior through the BI-RADS category it assigns.

This matters for the graph question too. Findings and lesions form a bipartite
graph via the appearance edges, so a network over it has something real to pass:
finding evidence informs which lesion is present, and a lesion hypothesis predicts
which other findings should also hold. That is a genuine second-order structure,
unlike the claim attack graph, which was measured to be independent of correctness.

The known asymmetry, recorded rather than papered over: this ontology catalogues
benign entities by appearance and describes malignancy through finding stances, so
the lesion layer is overwhelmingly benign. A lesion probe is therefore mostly a
BENIGNITY detector, and is expected to work by ruling out rather than ruling in.
"""
from __future__ import annotations

import json, re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_KG = _HERE.parents[1] / "breastMnist/data/breast/knowledge_graph.json"

# BI-RADS category -> malignancy prior, midpoint of the ACR likelihood band.
_BIRADS_PRIOR = {"birads_1": 0.0, "birads_2": 0.0, "birads_3": 0.01,
                 "birads_4": 0.485, "birads_5": 0.975}

# Near-synonyms the graph carries separately. Probing both would spend calls on
# the same question and double-weight it in any pooled readout.
_ALIAS = {"simple_cyst": "simple_breast_cyst",
          "complicated_cyst": "complicated_breast_cyst",
          "lipoma_of_breast": "breast_lipoma",
          "fat_necrosis_acute_phase": "fat_necrosis",
          "fat_necrosis_subacute_phase": "fat_necrosis",
          "fat_necrosis_late_phase": "fat_necrosis",
          "focal_mammary_fat_necrosis": "fat_necrosis",
          "intramammary_lymph_node_with_fat_hilum": "intramammary_lymph_node",
          "breast_implants": "implants"}

# Entities that are appearance phrases rather than nameable diagnoses.
_NOT_A_DIAGNOSIS = re.compile(
    r"^(hypoechoic|hyperechoic|microlobulated|refraction|complicated_cyst_with|"
    r"architectural_distortion_due_to|mixed_density|secretory|calcified_)")

# The two malignant entities the graph does carry. They have no `is_a` and no
# BI-RADS edge -- that absence IS the one-sided-ontology finding -- so their
# priors are asserted here explicitly rather than silently derived from nothing.
_MALIGNANT = {"breast_cancer": 0.975, "metastatic_intramammary_lymph_node": 0.975,
              "breast_neoplasms": 0.75}


def _plain(name):
    """`simple_breast_cyst` -> `a simple breast cyst`."""
    s = name.replace("_", " ")
    return ("an " if s[0] in "aeiou" else "a ") + s


def lesion_table():
    """[(entity, plain-English name, malignancy prior, n appearance edges, source)].

    Priority for the malignancy prior: an explicit BI-RADS category, then the
    `is_a benign_breast_lesion` membership, then the asserted malignant set.
    """
    tr = json.loads(_KG.read_text())["triples"]
    appearance, klass, isa = {}, {}, {}
    for t in tr:
        s_, r, o = t["subject"], t["relation"], t["object"]
        if r == "ultrasound_appearance_includes":
            appearance.setdefault(s_, []).append(o)
        elif r == "typically_classified_as":
            klass[s_] = o
        elif r in ("is_a", "subtype_of"):
            isa.setdefault(s_, []).append(o)

    merged = {}
    for ent, feats in appearance.items():
        if _NOT_A_DIAGNOSIS.match(ent):
            continue
        canon = _ALIAS.get(ent, ent)
        merged.setdefault(canon, []).extend(feats)

    out = []
    for ent, feats in merged.items():
        cat = klass.get(ent)
        if cat in _BIRADS_PRIOR:
            pr, src = _BIRADS_PRIOR[cat], cat
        elif ent in _MALIGNANT:
            pr, src = _MALIGNANT[ent], "asserted_malignant"
        elif "benign_breast_lesion" in isa.get(ent, []):
            pr, src = 0.0, "is_a benign_breast_lesion"
        else:
            continue
        out.append((ent, _plain(ent), pr, len(set(feats)), src))
    out.sort(key=lambda r: (-r[2], r[0]))
    return out


def bipartite(lesions, findings):
    """L x F incidence via descriptor-token overlap.

    Exact entity matching gives ZERO edges: the graph's appearance vocabulary
    (`hyperechoic`, `well_circumscribed`, `posterior_shadowing`) and its finding
    vocabulary (`hyperechoic_mass`, `circumscribed_margin`,
    `posterior_shadowing_with_solid_irregular_mass`) differ in surface form while
    naming the same thing. This is the same token overlap that builds the 72-edge
    finding-finding adjacency, so the two graphs stay consistent.
    """
    import sys as _sys
    import numpy as np
    _sys.path.insert(0, str(_HERE.parents[0] / "15_kggnn"))
    from kg_graph import _tokens                                       # noqa: E402
    tr = json.loads(_KG.read_text())["triples"]
    app = {}
    for t in tr:
        if t["relation"] == "ultrasound_appearance_includes":
            app.setdefault(t["subject"], set()).update(_tokens(t["object"]))
        elif t["relation"] == "combination_indicates":
            app.setdefault(t["object"], set()).update(_tokens(t["subject"]))
    canon = {}
    for e, toks in app.items():
        canon.setdefault(_ALIAS.get(e, e), set()).update(toks)
    B = np.zeros((len(lesions), len(findings)))
    for i, row in enumerate(lesions):
        lt = canon.get(row[0], set())
        for j, f in enumerate(findings):
            sh = lt & _tokens(f)
            if sh:
                B[i, j] = len(sh)
    return B


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_HERE.parents[0] / "14_kgtensor"))
    from claims import kg_findings
    L = lesion_table()
    print(f"{len(L)} probeable lesions\n")
    print(f"{'entity':40s} {'prior':>6s} {'app':>4s}  source")
    for ent, plain, pr, n, cat in L:
        print(f"{ent:40s} {pr:6.3f} {n:4d}  {cat}")
    names = [f for f, _ in kg_findings()]
    B = bipartite(L, names)
    print(f"\nbipartite lesion-finding incidence: {int(B.sum())} edges "
          f"over {B.shape[0]}x{B.shape[1]}")
    print("findings covered by at least one lesion: "
          f"{int((B.sum(0) > 0).sum())}/{len(names)}")
    for j, f in enumerate(names):
        if B[:, j].sum() == 0:
            print(f"   uncovered: {f}")
