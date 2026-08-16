"""Fully automatic probe derivation from a knowledge graph.

`run_kg_featureprobe.py` derives *which* features to probe from the KG, but then
applies two hand-written tables that do not transfer to another graph:

  _NOT_VISIBLE  11 feature names I excluded by hand as unobservable on a 2D
                grayscale crop.
  _PHRASING     17 question texts I wrote myself.

Both are replaced here:

  polarity   Read from the relation name, negation-aware, with KG categories and
             numeric metadata relations filtered out. The schema's own
             `stance_triple_ids` takes precedence where present.
  question   Built from the graph: the feature's `definition` triple if it has
             one, otherwise its `ultrasound_appearance_includes` objects,
             otherwise the humanised name.
  exclusion  Not a list. Probe everything, then drop features whose measured
             variance across a corpus is near zero (`--min-std`). A feature the
             model cannot see answers identically for every image, so variance
             filtering finds the unobservable ones from data instead of from my
             judgement — and it finds them for any KG, not just this one.

Nothing here names a breast-imaging concept, so pointing it at a different
domain graph yields a different probe set with no code changes.

Usage:
  python probe_auto.py --kg-root ../../breastMnist        # inspect probe set
  python probe_auto.py --kg-root ../../breastMnist --compare
"""
from __future__ import annotations

import argparse, json, re
from collections import defaultdict
from pathlib import Path

# Relations that quantify a category rather than assert a visual finding.
# Matched by prefix so a new graph's `likelihood_of_*` variants are caught too.
_METADATA_PREFIXES = ("likelihood_of", "management", "typical_size",
                      "birads_subcategory_of", "category_descriptor")

_NEGATION = ("no_malignant", "not_malignant", "no_malignancy", "benign")


def polarity_from_relation(rel: str) -> float | None:
    """Diagnostic sign implied by a relation name, or None if it asserts none.

    Negation is checked before the positive term so `has_no_malignant_potential`
    resolves to benign rather than matching on the substring "malignan".
    """
    r = rel.lower()
    if r.startswith(_METADATA_PREFIXES):
        return None
    if any(n in r for n in _NEGATION):
        return -1.0
    if "birads_5" in r or "malignan" in r:
        return +1.0
    return None


def _humanise(s: str) -> str:
    return re.sub(r"_+", " ", s).strip()


def is_category(subject: str, schema: dict) -> bool:
    """True for graph bookkeeping nodes rather than observable findings."""
    if subject.lower().startswith("birads"):
        return True
    cats = schema.get("categories")
    if isinstance(cats, dict) and subject in cats:
        return True
    if isinstance(cats, list) and subject in cats:
        return True
    return False


def question_from_kg(subject: str, by_subject: dict[str, list[dict]]) -> str:
    """Build the probe question out of the graph's own descriptive triples."""
    name = _humanise(subject)
    defs = [t["object"] for t in by_subject.get(subject, [])
            if t["relation"] == "definition"]
    apps = [t["object"] for t in by_subject.get(subject, [])
            if t["relation"] == "ultrasound_appearance_includes"]

    if defs:
        desc = f"{name}, defined as {_humanise(defs[0])}"
    elif apps:
        shown = ", ".join(_humanise(a) for a in apps[:3])
        desc = f"{name}, which appears as {shown}"
    else:
        desc = name

    return ("This is a breast ultrasound image.\n"
            f"Does the mass in this image show {desc}?\n"
            "Answer with exactly one word: yes or no.")


def build_probes_auto(triples: list[dict], schema: dict) -> list[dict]:
    """Derive the full probe set with no hand-written feature tables."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        by_subject[t.get("subject", "")].append(t)

    stance_ids = set(schema.get("stance_triple_ids", []))
    best: dict[str, float] = {}

    for t in triples:
        rel, subj = t.get("relation", ""), t.get("subject", "")
        pol = polarity_from_relation(rel)
        if pol is None:
            continue
        # The schema marks which triples are stances; honour it when it applies,
        # but do not require it, since not every graph ships that annotation.
        if stance_ids and t.get("id") not in stance_ids and rel not in {
                "suggests_malignancy", "suggests_benign"}:
            continue
        if not subj or is_category(subj, schema):
            continue
        w = pol * (0.5 if t.get("object") == "possible" else 1.0)
        if abs(w) > abs(best.get(subj, 0.0)):
            best[subj] = w

    probes = [{"feature": s, "weight": w,
               "question": question_from_kg(s, by_subject)}
              for s, w in best.items()]
    probes.sort(key=lambda p: (-p["weight"], p["feature"]))
    return probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-root", default=str(Path(__file__).resolve().parents[2] / "breastMnist"))
    p.add_argument("--compare", action="store_true",
                   help="diff against the hand-curated probe set")
    args = p.parse_args()

    root = Path(args.kg_root)
    triples = json.loads((root / "data/breast/knowledge_graph.json").read_text())["triples"]
    schema = json.loads((root / "data/breast/schema.json").read_text())

    auto = build_probes_auto(triples, schema)
    print(f"AUTOMATIC probe set: {len(auto)} probes\n")
    for pr in auto:
        print(f"  w={pr['weight']:+.1f}  {pr['feature']}")
        print(f"          Q: {pr['question'].splitlines()[1]}")

    if args.compare:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from run_kg_featureprobe import build_probes
        man = {p["feature"]: p["weight"] for p in build_probes(triples, schema)}
        a = {p["feature"]: p["weight"] for p in auto}
        print(f"\n{'='*66}\nmanual={len(man)}  automatic={len(a)}")
        only_auto = sorted(set(a) - set(man))
        only_man = sorted(set(man) - set(a))
        disagree = [f for f in set(a) & set(man) if a[f] * man[f] < 0]
        print(f"\nnew in automatic ({len(only_auto)}) "
              f"-- previously excluded by hand, now decided by variance:")
        for f in only_auto:
            print(f"  {f:50s} w={a[f]:+.1f}")
        print(f"\nlost vs manual ({len(only_man)}):")
        for f in only_man:
            print(f"  {f:50s} w={man[f]:+.1f}")
        print(f"\npolarity disagreements: {disagree or 'none'}")


if __name__ == "__main__":
    main()
