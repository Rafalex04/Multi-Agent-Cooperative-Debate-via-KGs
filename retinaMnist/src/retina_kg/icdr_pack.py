"""ICDR ontology -> the data pack every runner in this project already reads.

The two uploaded files are close to the repo's KG format but not identical, and
the gaps matter:

  * `icdr_definitions.json` entities carry no `id`. `kg/loader.py:78` does
    `entry["id"]` unguarded, so loading them as-is is a KeyError. We assign
    d_001..d_042 in source order (which is meaningful: levels first, then lesions).
  * `icdr_triples.json` carries a top-level `criteria` array with the quantitative
    thresholds (">=20 intraretinal haemorrhages in each quadrant"). BI-RADS has no
    analogue -- its thresholds are opaque strings -- and `load_kg_from_json` reads
    only `data["triples"]`, so criteria would be silently dropped. They are written
    to their own file and used as adjacency mediators.
  * triple ids are already t_001..t_152, and the extra fields (`axis`, `source`,
    `specificity`, `confidence`, `requires_criterion`) are a superset of what the
    loader needs, so they survive untouched.

FINDINGS. The breast pipeline derived its 16 findings through `build_probes`,
which leaned on three hand-written tables (`_POLARITY`, `_NOT_VISIBLE`,
`_PHRASING`) -- and EXPERIMENT_QUEUE.md records that automating them cost
0.6685 -> 0.5750. ICDR is in better shape: it states its own severity mapping,
its own observability constraints (`visible_on`, `located_in`), and carries a
full clinical definition for all 42 entities. So every table is derived from the
graph here, and `ontology_retina.yaml` holds the relation names.

  python icdr_pack.py                       # build data/retina/
  python icdr_pack.py --print-findings      # inspect what was derived
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import paths as P                                                    # noqa: E402


# ------------------------------------------------------------------ utilities --
def humanise(term):
    return str(term).replace("_", " ").strip()


def clean_sentences(text, leak_terms, n=2, limit=280):
    """Lead of a definition with every grading sentence removed.

    Returns "" when the source describes only what the lesion MEANS and never
    what it looks like -- which is true of 8 of the 17 ICDR findings, and is
    itself worth reporting rather than papering over.
    """
    text = " ".join(str(text).split())
    keep = [p for p in re.split(r"(?<=[.!?])\s+", text)
            if p and not any(w in p.lower() for w in leak_terms)]
    out = " ".join(keep[:n]).strip()
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0] + "..."
    return out


def _index(triples):
    by_rel = {}
    for t in triples:
        by_rel.setdefault(str(t["relation"]), []).append(t)
    return by_rel


# ------------------------------------------------------------------ derivation --
def derive_findings(triples, entities, cfg):
    """[{feature, description, level, prior, source_relation, ...}] from the KG.

    A finding is an entity that a level relation points FROM, is not itself a
    level, and that the ontology does not mark as unobservable on a monoscopic
    colour fundus photograph.
    """
    axis = cfg["axis"]
    levels = list(cfg["levels"])
    lvl_idx = {n: i for i, n in enumerate(levels)}
    by_rel = _index(triples)
    defs = {e["term"]: e.get("definition", "") for e in entities}
    leak = [w.lower() for w in cfg["description_leak_terms"]]

    # -- observability, straight out of the graph ------------------------------
    blocked, why = set(), {}
    for t in by_rel.get(cfg["visibility_relation"], []):
        if str(t["object"]) in set(cfg["unavailable_modalities"]):
            blocked.add(str(t["subject"]))
            why[str(t["subject"])] = f"{cfg['visibility_relation']} {t['object']}"
    for rel in cfg["location_relations"]:
        for t in by_rel.get(rel, []):
            if str(t["object"]) in set(cfg["excluded_locations"]):
                blocked.add(str(t["subject"]))
                why[str(t["subject"])] = f"{rel} {t['object']}"
    # `hypertensive_retinopathy mimics cotton_wool_spot` -- the SUBJECT is the
    # non-DR disease. Reading the object instead drops cotton wool spot, which
    # the ICDR table names explicitly in its moderate NPDR criterion.
    non_dr = {str(t["subject"]) for r in cfg["non_dr_marker_relations"]
              for t in by_rel.get(r, [])}
    # A finding must be a DEFINED entity. This drops the criterion sentinels
    # (`no_abnormalities`, the level-0 absence rule) and the composite shorthands
    # (`dot_and_blot_hemorrhage`) that exist only as triple subjects.
    defined = set(defs)

    # -- level assignment, most specific relation wins --------------------------
    assigned = {}
    for rel in cfg["level_relations"]:
        for t in by_rel.get(rel, []):
            if str(t.get("axis")) != axis:
                continue
            subj, obj = str(t["subject"]), str(t["object"])
            if obj not in lvl_idx or subj in lvl_idx:
                continue
            assigned.setdefault(subj, (lvl_idx[obj], rel, t["id"]))

    # -- appearance text, then definition lead, then the bare name --------------
    placeholder = cfg["appearance_placeholder"]
    appearance = {}
    for rel in cfg["appearance_relations"]:
        for t in by_rel.get(rel, []):
            if str(t["object"]) == placeholder:
                continue
            appearance.setdefault(str(t["subject"]), []).append(humanise(t["object"]))

    centre, scale = cfg["prior_centre"], cfg["prior_scale"]
    out, dropped = [], []
    for feat, (lvl, rel, tid) in sorted(assigned.items(), key=lambda kv: (kv[1][0], kv[0])):
        if feat in blocked or feat in non_dr or feat not in defined:
            reason = (why.get(feat) or ("not a defined entity" if feat not in defined
                                        else "differential/mimic of DR"))
            dropped.append({"feature": feat, "level": lvl, "reason": reason})
            continue
        clean = clean_sentences(defs.get(feat, ""), leak)
        if feat in appearance:
            desc, src = ", ".join(appearance[feat]), "appearance_triples"
        elif clean:
            desc, src = clean, "definition"
        else:
            desc, src = humanise(feat), "term_name"
        out.append({
            "feature": feat,
            "description": desc,
            "description_source": src,
            "level": lvl,
            "level_name": levels[lvl],
            "level_relation": rel,
            "level_triple": tid,
            "prior": round((lvl - centre) / scale, 6),
        })
    return out, dropped


# ----------------------------------------------------------------------- build --
def build(src_dir, out_dir, cfg):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_defs = json.loads((src_dir / "icdr_definitions.json").read_text())
    raw_tri = json.loads((src_dir / "icdr_triples.json").read_text())

    entities = []
    for i, e in enumerate(raw_defs["entities"], start=1):
        entities.append({**e, "id": f"d_{i:03d}"})
    relations = raw_defs["relations"]
    triples = raw_tri["triples"]

    (out_dir / "knowledge_graph.json").write_text(
        json.dumps({"triples": triples}, indent=1))
    (out_dir / "definitions.json").write_text(
        json.dumps({"entities": entities, "relations": relations}, indent=1))
    (out_dir / "criteria.json").write_text(json.dumps(
        {"criteria": raw_tri["criteria"],
         "extraction_notes": raw_tri["extraction_notes"]}, indent=1))

    findings, dropped = derive_findings(triples, entities, cfg)
    if len(findings) < cfg["min_findings"]:
        raise SystemExit(f"only {len(findings)} findings derived, "
                         f"minimum {cfg['min_findings']} -- relation names are wrong")
    (out_dir / "findings.json").write_text(json.dumps(
        {"findings": findings, "dropped": dropped,
         "levels": cfg["levels"], "axis": cfg["axis"],
         "prior_centre": cfg["prior_centre"], "prior_scale": cfg["prior_scale"]},
        indent=1))

    return {"n_triples": len(triples), "n_entities": len(entities),
            "n_criteria": len(raw_tri["criteria"]),
            "n_findings": len(findings), "n_dropped": len(dropped),
            "out_dir": str(out_dir)}


def load_findings(pack_dir):
    """[(feature, description, level, prior)] -- the retina analogue of kg_findings()."""
    d = json.loads((Path(pack_dir) / "findings.json").read_text())
    return [(f["feature"], f["description"], f["level"], f["prior"])
            for f in d["findings"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(P.SOURCE))
    ap.add_argument("--out-dir", default=str(P.PACK))
    ap.add_argument("--ontology", default=str(P.ONTOLOGY))
    ap.add_argument("--print-findings", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.ontology).read_text())
    rec = build(a.src, a.out_dir, cfg)
    print(json.dumps(rec, indent=1))

    if a.print_findings:
        d = json.loads((Path(a.out_dir) / "findings.json").read_text())
        print(f"\n{'feature':38s} {'lvl':4s} {'prior':7s} {'via':28s} desc-src")
        for f in d["findings"]:
            print(f"{f['feature']:38s} {f['level']:<4d} {f['prior']:+7.3f} "
                  f"{f['level_relation']:28s} {f['description_source']}")
        print(f"\ndropped ({len(d['dropped'])}):")
        for f in d["dropped"]:
            print(f"  {f['feature']:36s} lvl {f['level']}  {f['reason']}")


if __name__ == "__main__":
    main()
