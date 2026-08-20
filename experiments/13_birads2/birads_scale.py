"""The BI-RADS ladder, read out of the knowledge graph rather than written down.

Every number this experiment depends on -- which categories exist, what each one
means, and what malignancy probability it stands for -- comes from triples the
graph already carried:

    birads_3   likelihood_of_malignancy_range   >0_to_<=2_percent
    birads_4b  likelihood_of_malignancy_range   >10_to_<=50_percent
    birads_5   likelihood_of_malignancy_minimum >=95_percent

That is the whole reason to ask for a category instead of a 0-10 score. A 0-10
scale needs an anchor written into the prompt by hand, and the mapping from
"7 out of 10" to a probability is invented. BI-RADS arrives with its own
calibration attached, so the KG converts a category into a probability and the
two turns can be averaged on a scale that means something.

Nothing here names a category literally. Point it at another graph with
likelihood triples and it builds that graph's ladder instead.
"""
from __future__ import annotations

import json, re
from pathlib import Path

_LIKELIHOOD = "likelihood_of_malignancy"      # matches ..._range / ..._minimum too
_SUBCAT     = "birads_subcategory_of"
_DEFINITION = "definition"
_PPV        = "ppv"        # a lesion's point predictive value, not a category band


def _midpoint(obj: str) -> float | None:
    """A KG likelihood object -> the midpoint of the interval it names, in percent.

    'essentially_0_percent' -> 0.0        '>0_to_<=2_percent'  -> 1.0
    '>10_to_<=50_percent'   -> 30.0       '>=95_percent'       -> 97.5
    An open upper bound is closed at 100, which is what '>=95' means in context.
    """
    s = str(obj)
    if "essentially_0" in s:
        return 0.0
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return None
    if s.startswith(">="):
        return (nums[0] + 100.0) / 2
    if len(nums) == 1:
        return nums[0] / 2
    return sum(nums[:2]) / 2


def _humanise(obj: str) -> str:
    return (str(obj).replace("_", " ")
            .replace("<=", "up to ").replace(">=", "at least ")
            .replace(">", "over ").strip())


def load_scale(kg_path: str | Path) -> list[dict]:
    """Ordered ladder of assessment categories, lowest malignancy risk first.

    Three structural filters, in order, none of which names a category:

    1. The entity carries a likelihood triple.  This alone is too loose: lesion
       *types* carry malignancy rates too (complex_cyst_type_i, 7.1 percent).
    2. That likelihood is an INTERVAL, not a point PPV.  This is the real
       distinction and not a cosmetic one -- an assessment category *is* a
       probability band ('>10 to <=50 percent'), whereas a lesion type has a
       single positive predictive value ('7.1_percent_ppv').  Averaging two
       turns is only meaningful on the band scale.
    3. The entity is not `is_a`-typed, which is how the graph marks lesions and
       descriptors and never marks an assessment category.

    Finally, a category that others declare themselves subcategories of is
    dropped as redundant: that removes the unqualified birads_4 in favour of
    4a/4b/4c, whose bands partition it.
    """
    triples = json.loads(Path(kg_path).read_text())["triples"]
    is_typed = {str(t["subject"]) for t in triples
                if str(t.get("relation")) == "is_a"}
    parents = {str(t["object"]) for t in triples
               if str(t.get("relation")) == _SUBCAT}
    child_of = {str(t.get("subject")): str(t.get("object")) for t in triples
                if str(t.get("relation")) == _SUBCAT}
    defn = {str(t["subject"]): _humanise(t["object"]) for t in triples
            if str(t.get("relation")) == _DEFINITION}

    out = []
    for t in triples:
        s, r, o = str(t.get("subject", "")), str(t.get("relation", "")), str(t.get("object"))
        if not r.startswith(_LIKELIHOOD):
            continue
        if _PPV in o.lower():          # filter 2: point estimate, not a band
            continue
        if s in is_typed or s in parents:
            continue
        p = _midpoint(o)
        if p is None:
            continue
        out.append({"name": s,
                    "code": s.split("_", 1)[1] if "_" in s else s,
                    "p_mal": p,
                    "range": _humanise(o),
                    "label": defn.get(s) or defn.get(child_of.get(s, ""), "")})
    out.sort(key=lambda d: (d["p_mal"], d["code"]))
    for i, d in enumerate(out):
        d["ordinal"] = i
    return out


def menu(scale: list[dict]) -> str:
    """The category list as it appears in the prompt, built from the ladder."""
    w = max(len(d["code"]) for d in scale)
    return "\n".join(f"  {d['code']:<{w}} = {d['label']}, {d['range']} likelihood of malignancy"
                     for d in scale)


_CODE_RE = re.compile(r"\b([0-6])\s*([abc])?\b", re.I)


def parse_category(text: str, scale: list[dict]) -> dict | None:
    """First BI-RADS code in the model's reply, resolved against the ladder.

    An unqualified '4' is not on the ladder (4a/4b/4c replaced it), but the graph
    still defines it, so it resolves to the parent's own midpoint via `extra`.
    """
    by_code = {d["code"]: d for d in scale}
    for m in _CODE_RE.finditer(text or ""):
        code = m.group(1) + (m.group(2) or "").lower()
        if code in by_code:
            return by_code[code]
    return None


def nearest(p: float, scale: list[dict]) -> dict:
    """The ladder rung closest to a probability -- turns an average back into a verdict."""
    return min(scale, key=lambda d: abs(d["p_mal"] - p))


if __name__ == "__main__":
    import sys
    sc = load_scale(sys.argv[1] if len(sys.argv) > 1
                    else "breastMnist/data/breast/knowledge_graph.json")
    print(f"{len(sc)} categories on the ladder\n")
    for d in sc:
        print(f"  ord {d['ordinal']}  BI-RADS {d['code']:<3s} p_mal {d['p_mal']:5.1f}%  "
              f"{d['label']} ({d['range']})")
    print("\n--- prompt menu ---")
    print(menu(sc))
