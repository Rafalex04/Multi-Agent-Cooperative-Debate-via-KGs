"""Dermoscopy ontology -> the data pack every runner in this project reads.

Mirrors retinaMnist/src/retina_kg/icdr_pack.py, with one structural difference
that follows from the label space:

  ICDR   finding --level_relation--> level        ordinal, prior is a SCALAR
  Derma  finding --establishes_lesion_class--> lesion --maps_to--> class
                                                nominal, prior is a 7-VECTOR

So a derma finding's prior is a distribution over the seven DermaMNIST classes,
normalised, rather than a position on a ladder. `nominal.py` consumes it as the
L2 penalty centre for the class-specific readout.

As on retina, every table the breast pipeline hand-wrote is derived from the
graph: which findings exist, what they indicate, which are unobservable, and how
they are described. `ontology_derma.yaml` holds the relation names.

  python derma_pack.py --print-findings
"""
from __future__ import annotations

import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import paths as P                                                    # noqa: E402


def humanise(term):
    return str(term).replace("_", " ").strip()


def clean_sentences(text, leak_terms, n=2, limit=280):
    """Definition lead with every diagnosis-naming sentence removed."""
    text = " ".join(str(text).split())
    keep = [p for p in re.split(r"(?<=[.!?])\s+", text)
            if p and not any(w in p.lower() for w in leak_terms)]
    out = " ".join(keep[:n]).strip()
    if len(out) > limit:
        out = out[:limit].rsplit(" ", 1)[0] + "..."
    return out


def _by_rel(triples):
    d = defaultdict(list)
    for t in triples:
        d[str(t["relation"])].append(t)
    return d


def derive_findings(triples, entities, cfg):
    """[{feature, description, classes, prior[7], ...}] straight from the graph."""
    classes = list(cfg["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    by = _by_rel(triples)
    defs = {e["term"]: e.get("definition", "") for e in entities}
    leak = [w.lower() for w in cfg["description_leak_terms"]]

    # LESION entity -> DermaMNIST class. Only maps_to_dermamnist_class defines
    # this. The per-triple `dermamnist_class` field is NOT an entity mapping --
    # it annotates which class a given triple is about, and its subject is
    # usually the finding, not the lesion. Conflating the two put every finding
    # into ent_class and then dropped them all as "lesion entities".
    ent_class = {str(t["subject"]): str(t["object"])
                 for t in by.get(cfg["class_relation"], [])
                 if str(t["object"]) in cidx}

    # finding -> classes it supports. Three sources, most specific first:
    #   1. the triple's own dermamnist_class field (187 of 302 carry it)
    #   2. the object resolved through ent_class (the two-hop path)
    #   3. the object already being a class name
    support = defaultdict(set)
    via = {}
    for rel in cfg["finding_relations"]:
        for t in by.get(rel, []):
            f, o = str(t["subject"]), str(t["object"])
            cls = (t.get("dermamnist_class") if t.get("dermamnist_class") in cidx
                   else ent_class.get(o) or (o if o in cidx else None))
            if cls is None:
                continue
            support[f].add(cls)
            via.setdefault(f, (rel, t["id"]))

    # observability, from the graph
    blocked, why = {}, {}
    for t in by.get(cfg["visibility_relation"], []):
        if str(t["object"]) in set(cfg["unavailable_modalities"]):
            blocked[str(t["subject"])] = f"{cfg['visibility_relation']} {t['object']}"
    for rel in cfg["exclusion_relations"]:
        for t in by.get(rel, []):
            blocked.setdefault(str(t["subject"]), f"{rel} {t['object']}")

    appearance = defaultdict(list)
    placeholder = cfg["appearance_placeholder"]
    for rel in cfg["appearance_relations"]:
        for t in by.get(rel, []):
            if str(t["object"]) != placeholder:
                appearance[str(t["subject"])].append(humanise(t["object"]))

    defined = set(defs)
    out, dropped = [], []
    for feat in sorted(support):
        cls = sorted(support[feat])
        if feat in blocked:
            dropped.append({"feature": feat, "reason": blocked[feat]}); continue
        # NOT dropped for lacking a definition entry. On ICDR that rule removed
        # criterion sentinels; here it removed `central_scar_like_area` (the
        # dermatofibroma sign) and every `*_lacunae` (vascular), leaving df with
        # ZERO findings and vasc with one. An undefined finding simply falls back
        # to its term name as a description, which is already the fallback path.
        if feat in ent_class:      # it IS a lesion class, not an observation of one
            dropped.append({"feature": feat, "reason": f"is a lesion entity ({ent_class[feat]})"})
            continue
        clean = clean_sentences(defs.get(feat, ""), leak)
        if feat in appearance:
            desc, src = ", ".join(appearance[feat]), "appearance_triples"
        elif clean:
            desc, src = clean, "definition"
        else:
            desc, src = humanise(feat), "term_name"
        prior = [0.0] * len(classes)
        for c in cls:
            prior[cidx[c]] = 1.0 / len(cls)
        rel, tid = via.get(feat, (None, None))
        out.append({"feature": feat, "description": desc, "description_source": src,
                    "classes": cls, "prior": prior,
                    "n_classes_supported": len(cls),
                    "via_relation": rel, "via_triple": tid})
    return out, dropped


def build(src_dir, out_dir, cfg):
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = json.loads((src_dir / "derma_definitions.json").read_text())
    rt = json.loads((src_dir / "derma_triples.json").read_text())

    entities = [{**e, "id": f"d_{i:03d}"} for i, e in enumerate(rd["entities"], 1)]
    (out_dir / "knowledge_graph.json").write_text(
        json.dumps({"triples": rt["triples"]}, indent=1))
    (out_dir / "definitions.json").write_text(
        json.dumps({"entities": entities, "relations": rd["relations"],
                    "vocabulary_notes": rd.get("vocabulary_notes", [])}, indent=1))
    (out_dir / "criteria.json").write_text(json.dumps(
        {"criteria": rt["criteria"], "gaps": rt.get("gaps", []),
         "source_document": rt.get("source_document", "")}, indent=1))

    findings, dropped = derive_findings(rt["triples"], entities, cfg)
    if len(findings) < cfg["min_findings"]:
        raise SystemExit(f"only {len(findings)} findings derived, minimum "
                         f"{cfg['min_findings']} -- relation names are wrong")
    (out_dir / "findings.json").write_text(json.dumps(
        {"findings": findings, "dropped": dropped, "classes": cfg["classes"],
         "class_names": cfg["class_names"], "task": cfg["task"]}, indent=1))
    (out_dir / "probe_findings.json").write_text(json.dumps(
        [[f["feature"], f["description"], ",".join(f["classes"])] for f in findings],
        indent=1))
    (out_dir / "lexicon.json").write_text(json.dumps({
        "head": "You are an experienced dermatologist examining a dermoscopic image.\n",
        "subject": "this lesion", "subject_cap": "This lesion",
        "peer": "dermatologist"}, indent=1))
    return {"n_triples": len(rt["triples"]), "n_entities": len(entities),
            "n_criteria": len(rt["criteria"]), "n_findings": len(findings),
            "n_dropped": len(dropped), "out_dir": str(out_dir)}


def load_findings(pack_dir):
    d = json.loads((Path(pack_dir) / "findings.json").read_text())
    return [(f["feature"], f["description"], f["classes"], f["prior"])
            for f in d["findings"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(P.SOURCE))
    ap.add_argument("--out-dir", default=str(P.PACK))
    ap.add_argument("--ontology", default=str(P.ONTOLOGY))
    ap.add_argument("--print-findings", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.ontology).read_text())
    print(json.dumps(build(a.src, a.out_dir, cfg), indent=1))
    if a.print_findings:
        d = json.loads((Path(a.out_dir) / "findings.json").read_text())
        print(f"\n{'feature':40s} {'classes':22s} {'via':26s} desc-src")
        for f in d["findings"]:
            print(f"{f['feature']:40s} {','.join(f['classes']):22s} "
                  f"{str(f['via_relation']):26s} {f['description_source']}")
        print(f"\ndropped ({len(d['dropped'])}):")
        for f in d["dropped"][:15]:
            print(f"  {f['feature']:38s} {f['reason']}")
if __name__ == "__main__":
    main()
