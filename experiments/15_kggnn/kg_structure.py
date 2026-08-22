"""Data contract: the ontology's finding / lesion / class structure.

This is what S1 and S6 consume. Three pieces, all read from the graph:

  findings   the 16 debate findings and the KG's stance sign for each
  lesions    entities the ontology describes by the findings they PRESENT,
             with the Adamic-Adar weight of each (lesion, finding) pair
  classes    each lesion's malignancy class

The class assignment is the part that has to be careful. A keyword scan over
everything a lesion is connected to gets it wrong: `fibroadenoma mimics
phyllodes_tumour` makes a benign lesion look malignant, and `intramammary_lymph_node
mimics early_breast_cancer_rarely` does the same. `mimics` and `differential_for`
describe what a lesion can be CONFUSED WITH, which is close to the opposite of
what it IS.

So class comes only from relations that actually classify:

    is_a  ... benign_breast_lesion            -> benign
    typically_classified_as  birads_1/2/3     -> benign      (<= 2 percent)
    typically_classified_as  birads_4/5       -> malignant   (>  2 percent)

with the BI-RADS threshold read from the graph's own likelihood ladder rather
than hardcoded. A lesion the graph does not classify is dropped: it cannot
contribute to a fixed lesion->class readout, and guessing would put noise into
the one part of the architecture that is supposed to be given.
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "13_birads2"))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from kg_graph import _tokens, lesion_descriptions                     # noqa: E402

_CLASSIFY_ISA = "is_a"
_CLASSIFY_BR = "typically_classified_as"
_BENIGN_ISA = "benign_breast_lesion"
# A lesion the graph lists malignancy PREDICTORS for is on the suspicious side.
# This is a classifying statement, unlike `mimics`, which says only what the
# lesion can be confused with.
_CLASSIFY_PRED = "associated_with_malignancy_predictor"
_CLASSIFY_RISK = "likelihood_of_malignancy"


def _birads_risk(kg_path):
    """BI-RADS category -> midpoint malignancy probability, from the graph."""
    from birads_scale import load_scale
    return {d["code"]: d["p_mal"] for d in load_scale(kg_path)}


def build(kg_path, findings, risk_cut=2.0):
    """(findings, lesions, mask) contract dict."""
    triples = json.loads(Path(kg_path).read_text())["triples"]
    names = [f for f, _ in findings]
    fset = set(names)
    risk = _birads_risk(kg_path)

    # ---- lesion -> findings it presents, via the appearance descriptions ----
    desc = lesion_descriptions(triples)
    presents = defaultdict(set)
    for i, f in enumerate(names):
        tf = _tokens(f)
        for les, dt in desc.items():
            if les in fset:            # a finding describing itself is not a lesion
                continue
            if tf & dt:
                presents[les].add(i)

    # ---- class, from classifying relations only ----
    cls = {}
    for t in triples:
        s, r, o = str(t.get("subject")), str(t.get("relation")), str(t.get("object"))
        if s not in presents:
            continue
        if r == _CLASSIFY_PRED:
            cls[s] = "MAL"
        elif r.startswith(_CLASSIFY_RISK):
            from birads_scale import _midpoint
            p = _midpoint(o)
            if p is not None:
                cls[s] = "MAL" if p > risk_cut else "BEN"
        elif r == _CLASSIFY_ISA and o == _BENIGN_ISA:
            cls.setdefault(s, "BEN")
        elif r == _CLASSIFY_BR:
            code = o.replace("birads_", "")
            p = risk.get(code)
            if p is not None:
                # BI-RADS 3 and below is the "no biopsy" side of the ladder
                cls[s] = "MAL" if p > risk_cut else "BEN"

    lesions = sorted(l for l in presents if l in cls and presents[l])
    lidx = {l: i for i, l in enumerate(lesions)}

    # ---- Adamic-Adar weight per (lesion, finding) pair ----
    import numpy as np
    n_f_of = {l: len(presents[l]) for l in lesions}
    mask = np.zeros((len(lesions), len(names)))
    for l in lesions:
        for fi in presents[l]:
            mask[lidx[l], fi] = 1.0 / np.log1p(1 + n_f_of[l])
    if mask.max() > 0:
        mask /= mask.max()

    sign = np.array([1.0 if cls[l] == "MAL" else -1.0 for l in lesions])
    return {"findings": names,
            "finding_sign": [s for _, s in findings],
            "lesions": lesions,
            "lesion_class": [cls[l] for l in lesions],
            "lesion_sign": sign,
            "mask": mask,
            "presents": {l: sorted(presents[l]) for l in lesions},
            "dropped_unclassified": sorted(l for l in presents if l not in cls)}


if __name__ == "__main__":
    from claims import kg_findings
    F = kg_findings()
    kg = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
    c = build(kg, F)
    import numpy as np
    print(f"findings {len(c['findings'])}   lesions {len(c['lesions'])}   "
          f"(lesion, finding) edges {int((c['mask']>0).sum())}")
    nm=sum(1 for x in c['lesion_class'] if x=='MAL'); nb=len(c['lesion_class'])-nm
    print(f"lesion classes: MAL {nm}  BEN {nb}")
    if nm == 0 or nb == 0:
        print("  !! one-sided catalogue: the lesion layer is a single-class "
              "pattern detector, not a differential")
    print(f"dropped unclassified: {len(c['dropped_unclassified'])} {c['dropped_unclassified'][:6]}")
    print("\nlesion layer:")
    for i, l in enumerate(c["lesions"]):
        fs = [c["findings"][j][:26] for j in c["presents"][l]]
        print(f"  [{c['lesion_class'][i]}] {l:38s} <- {', '.join(fs[:4])}"
              + (f" (+{len(fs)-4})" if len(fs) > 4 else ""))
    cov = (c["mask"] > 0).sum(0)
    print("\nfindings with no lesion:",
          [c["findings"][j] for j in range(len(c["findings"])) if cov[j] == 0])
