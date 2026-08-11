"""Build the merged debate+KG graph dataset from completed debate JSON outputs.

Each sample becomes one heterogeneous graph:

  Node types
  ----------
  claim    — one DebateNode (a [CLAIM] block) from the structured debate.
  triple   — one ACR BI-RADS KG triple (subject–predicate–object) that is
             connected to at least one claim in this sample.

  Edge types
  ----------
  claim→claim   (AGREE / DISAGREE): parsed from [ADDRESSED:cN][AGREE|DISAGREE] tags.
  claim→triple  (retrieval):        top-k TF-IDF nearest KG triples per claim.
  triple→triple (shared_entity):    two triples sharing a subject or object entity.

Graph-level label: BENIGN=0, MALIGNANT=1 (gold label from BreastMNIST test set).

Output layout
-------------
  data/breast/dataset/
    graphs/sample_{id}.json   — one file per sample
    index.json                — {sample_id: {label, verdict, correct, ...}}
    stats.json                — aggregate dataset statistics

Usage
-----
  python -m debate_kg.dataset.build_graph_dataset \\
      --debate-dir outputs/2026-07-16/11-23-16 \\
      --out-dir    data/breast/dataset \\
      --k 3

The script is idempotent: if a sample graph already exists it is skipped
(--force to overwrite).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _triple_text(triple: object) -> str:
    return (
        f"{triple.subject.replace('_', ' ')} "
        f"{triple.predicate.replace('_', ' ')} "
        f"{triple.object.replace('_', ' ')}"
    )


def build_graph_for_sample(
    sample_id: str,
    debate: dict,
    kg,
    retriever,
    gold_label: str,
    k: int = 3,
) -> dict:
    """Return a dict representing the merged debate+KG graph for one sample."""
    nodes_raw = debate["nodes"]
    edges_raw = debate["edges"]

    # uuid → short_id mapping for resolving debate edges
    uuid_to_short = {n["id"]: n["short_id"] for n in nodes_raw}

    # 1. Claim nodes
    label_int = {"BENIGN": 0, "MALIGNANT": 1}
    claim_nodes = [
        {
            "node_id": n["short_id"],
            "uuid": n["id"],
            "type": "claim",
            "text": n["text"],
            "label": n.get("label"),
            "label_int": label_int.get(n.get("label"), 2),
            "expert_id": n["expert_id"],
            "round_idx": n["round_idx"],
        }
        for n in nodes_raw
    ]

    # 2. Re-retrieve top-k KG triples for each claim (ignore stored provenance)
    triple_by_uuid = {t.uuid: t for t in kg.all_triples()}
    claim_triple_edges: list[dict] = []
    referenced_uuids: set[str] = set()

    for n in nodes_raw:
        retrieved = retriever.retrieve(n["text"], k=k)
        for rank, triple in enumerate(retrieved):
            claim_triple_edges.append(
                {
                    "src": n["short_id"],
                    "dst": triple.uuid,
                    "rank": rank,
                    "type": "retrieval",
                }
            )
            referenced_uuids.add(triple.uuid)

    # 3. Triple nodes (only those referenced by at least one claim)
    triple_nodes = []
    for uid in sorted(referenced_uuids):
        t = triple_by_uuid.get(uid)
        if t is None:
            continue
        triple_nodes.append(
            {
                "node_id": uid,
                "uuid": uid,
                "type": "triple",
                "subject": t.subject.replace("_", " "),
                "predicate": t.predicate.replace("_", " "),
                "object": t.object.replace("_", " "),
                "text": _triple_text(t),
            }
        )

    # 4. Claim→claim edges (AGREE / DISAGREE)
    claim_claim_edges = []
    for e in edges_raw:
        src = uuid_to_short.get(e["source_id"], e["source_id"])
        dst = uuid_to_short.get(e["target_id"], e["target_id"])
        sign = e.get("sign", "+")
        claim_claim_edges.append(
            {
                "src": src,
                "dst": dst,
                "sign": sign,
                "type": "AGREE" if sign == "+" else "DISAGREE",
            }
        )

    # 5. Triple→triple edges (shared subject or object entity)
    ref_triples = [triple_by_uuid[uid] for uid in sorted(referenced_uuids) if uid in triple_by_uuid]
    triple_triple_edges: list[dict] = []
    for i, t1 in enumerate(ref_triples):
        for j, t2 in enumerate(ref_triples):
            if j <= i:
                continue
            shared = (
                t1.subject == t2.subject
                or t1.object == t2.object
                or t1.subject == t2.object
                or t1.object == t2.subject
            )
            if shared:
                triple_triple_edges.append(
                    {"src": t1.uuid, "dst": t2.uuid, "type": "shared_entity"}
                )

    rounds_used = max((n["round_idx"] for n in nodes_raw), default=0) + 1

    return {
        "sample_id": sample_id,
        "gold_label": gold_label,
        "gold_label_int": label_int.get(gold_label, -1),
        "verdict": debate.get("verdict"),
        "correct": debate.get("verdict") == gold_label,
        "rounds_used": rounds_used,
        "nodes": {
            "claims": claim_nodes,
            "triples": triple_nodes,
        },
        "edges": {
            "claim_claim": claim_claim_edges,
            "claim_triple": claim_triple_edges,
            "triple_triple": triple_triple_edges,
        },
        "stats": {
            "num_claims": len(claim_nodes),
            "num_triples": len(triple_nodes),
            "num_claim_claim_edges": len(claim_claim_edges),
            "num_claim_triple_edges": len(claim_triple_edges),
            "num_triple_triple_edges": len(triple_triple_edges),
        },
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_debate_dirs(base_outputs: Path) -> list[Path]:
    """Walk outputs/ looking for directories that contain debate_sample-*.json files."""
    dirs: set[Path] = set()
    for p in base_outputs.rglob("debate_sample-*.json"):
        dirs.add(p.parent)
    return sorted(dirs)


def _detect_dir_split(debate_dir: Path) -> str | None:
    """Return 'train', 'val', or 'test' from the .split_* marker file, or None."""
    for marker in debate_dir.glob(".split_*"):
        return marker.name.removeprefix(".split_")
    return None


def _load_all_gold_labels(data_dir: Path, breastmnist_root: str) -> dict[str, dict[str, str]]:
    """Load gold labels for all splits from all index files.

    Returns {split: {sample_id: gold_label}} for 'train', 'val', 'test'.
    """
    import numpy as np

    root = Path(breastmnist_root)
    npz_path = root / "breastmnist_224.npz"
    if not npz_path.exists():
        import medmnist
        npz_path = Path(medmnist.__file__).parent / "breastmnist_224.npz"
    data = np.load(str(npz_path))

    configs = [
        ("train", data_dir / "train_indices.json",     data["train_labels"]),
        ("val",   data_dir / "val_indices.json",        data["val_labels"]),
        ("test",  data_dir / "ablation_indices.json",   data["test_labels"]),
        ("test",  data_dir / "remaining_indices.json",  data["test_labels"]),
    ]

    result: dict[str, dict[str, str]] = {"train": {}, "val": {}, "test": {}}
    for split, path, labels_arr in configs:
        if not path.exists():
            logger.warning("Index file not found: %s", path)
            continue
        raw = json.loads(path.read_text())
        indices = raw["indices"] if isinstance(raw, dict) else raw
        for idx in indices:
            lbl = int(labels_arr[idx, 0])
            sample_id = f"{idx:03d}"
            # MedMNIST v2 BreastMNIST: 0 → MALIGNANT, 1 → BENIGN
            result[split][sample_id] = "MALIGNANT" if lbl == 0 else "BENIGN"
        logger.info("Loaded %d gold labels for split=%s from %s", len(indices), split, path.name)

    return result


def _find_debate_json(debate_dir: Path, sample_id: str) -> Path | None:
    candidate = debate_dir / f"debate_sample-{sample_id}.json"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(
    debate_dirs: list[Path],
    out_dir: Path,
    all_gold_labels: dict[str, dict[str, str]],
    k: int,
    force: bool,
    kg_path: Path,
    definitions_path: Path,
) -> dict:
    """Build graph JSON for every sample found across debate_dirs.

    Debate dirs are grouped by split using their .split_* marker files.
    Graphs are written to out_dir/{split}/graphs/sample_{id}.json.
    Returns a combined index over all splits.
    """
    from debate_kg.kg.loader import load_definitions, load_kg_from_json
    from debate_kg.retriever.tfidf_retriever import TFIDFRetriever

    kg = load_kg_from_json(kg_path)
    load_definitions(definitions_path)  # not used in builder but validates the file
    retriever = TFIDFRetriever(kg.all_triples())
    logger.info("Loaded KG: %d triples | TF-IDF retriever ready (k=%d)", len(kg), k)

    # Group debate dirs by split; dirs without a marker are skipped
    split_to_dirs: dict[str, dict[str, Path]] = {"train": {}, "val": {}, "test": {}}
    skipped_dirs = 0
    for d in debate_dirs:
        split = _detect_dir_split(d)
        if split is None:
            logger.warning("No .split_* marker in %s — skipping", d)
            skipped_dirs += 1
            continue
        if split not in split_to_dirs:
            logger.warning("Unknown split '%s' in %s — skipping", split, d)
            skipped_dirs += 1
            continue
        for p in sorted(d.glob("debate_sample-*.json")):
            sid = p.stem.replace("debate_sample-", "")
            split_to_dirs[split][sid] = p  # later dir wins if duplicate sample ID

    if skipped_dirs:
        logger.info("Skipped %d dirs with no split marker", skipped_dirs)

    combined_index: dict[str, dict] = {}

    for split in ("train", "val", "test"):
        sample_to_path = split_to_dirs[split]
        gold_labels = all_gold_labels.get(split, {})
        logger.info("=== Split=%s | %d debate files | %d gold labels ===",
                    split, len(sample_to_path), len(gold_labels))

        split_graphs_dir = out_dir / split / "graphs"
        split_graphs_dir.mkdir(parents=True, exist_ok=True)

        split_index: dict[str, dict] = {}

        for sample_id, debate_path in sorted(sample_to_path.items()):
            out_path = split_graphs_dir / f"sample_{sample_id}.json"

            if out_path.exists() and not force:
                logger.info("  [%s] sample %s — already built, skipping", split, sample_id)
                existing = json.loads(out_path.read_text())
                entry = {k2: existing[k2] for k2 in ("gold_label", "gold_label_int", "verdict", "correct", "rounds_used", "stats")}
                split_index[sample_id] = entry
                combined_index[f"{split}/{sample_id}"] = entry
                continue

            gold = gold_labels.get(sample_id)
            if gold is None:
                logger.warning("  [%s] sample %s — no gold label found, skipping", split, sample_id)
                continue

            debate = json.loads(debate_path.read_text())
            if not debate.get("nodes"):
                logger.warning("  [%s] sample %s — empty debate JSON, skipping", split, sample_id)
                continue

            graph = build_graph_for_sample(sample_id, debate, kg, retriever, gold, k=k)
            out_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False))

            s = graph["stats"]
            logger.info(
                "  [%s] sample %s (%s) — %d claims, %d triples | verdict=%s correct=%s",
                split, sample_id, gold,
                s["num_claims"], s["num_triples"],
                graph["verdict"], graph["correct"],
            )
            entry = {
                "gold_label": graph["gold_label"],
                "gold_label_int": graph["gold_label_int"],
                "verdict": graph["verdict"],
                "correct": graph["correct"],
                "rounds_used": graph["rounds_used"],
                "stats": graph["stats"],
            }
            split_index[sample_id] = entry
            combined_index[f"{split}/{sample_id}"] = entry

        # Write per-split index and stats
        (out_dir / split / "index.json").write_text(json.dumps(split_index, indent=2))
        split_stats = _compute_stats(split_index)
        split_stats["split"] = split
        (out_dir / split / "stats.json").write_text(json.dumps(split_stats, indent=2))
        logger.info("  Split %s done: %d graphs | acc=%.4f", split,
                    split_stats.get("num_samples", 0), split_stats.get("accuracy", 0))

    return combined_index


def _compute_stats(index: dict) -> dict:
    if not index:
        return {}
    items = list(index.values())
    n = len(items)
    correct = sum(1 for x in items if x["correct"])
    mal = sum(1 for x in items if x["gold_label"] == "MALIGNANT")
    total_claims = sum(x["stats"]["num_claims"] for x in items)
    total_triples = sum(x["stats"]["num_triples"] for x in items)
    total_cc = sum(x["stats"]["num_claim_claim_edges"] for x in items)
    total_ct = sum(x["stats"]["num_claim_triple_edges"] for x in items)
    total_tt = sum(x["stats"]["num_triple_triple_edges"] for x in items)
    return {
        "num_samples": n,
        "num_malignant": mal,
        "num_benign": n - mal,
        "accuracy": round(correct / n, 4) if n else 0,
        "avg_claims_per_graph": round(total_claims / n, 2),
        "avg_triples_per_graph": round(total_triples / n, 2),
        "avg_claim_claim_edges": round(total_cc / n, 2),
        "avg_claim_triple_edges": round(total_ct / n, 2),
        "avg_triple_triple_edges": round(total_tt / n, 2),
        "total_claim_nodes": total_claims,
        "total_triple_nodes": total_triples,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build merged debate+KG graph dataset")
    parser.add_argument("--debate-dir", action="append", dest="debate_dirs",
                        help="Directory containing debate_sample-*.json files (repeatable)")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover all debate dirs under outputs/")
    parser.add_argument("--out-dir", default="data/breast/dataset",
                        help="Output directory for graphs/ index.json stats.json")
    parser.add_argument("--k", type=int, default=3,
                        help="Top-k KG triples per claim (default: 3)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing sample graphs")
    parser.add_argument("--kg-path", default="data/breast/knowledge_graph.json")
    parser.add_argument("--definitions-path", default="data/breast/definitions.json")
    parser.add_argument("--data-dir", default="data/breast",
                        help="Directory containing *_indices.json files (default: data/breast)")
    parser.add_argument("--breastmnist-root", default=None,
                        help="Root dir for breastmnist_224.npz (defaults to medmnist cache)")
    args = parser.parse_args(argv)

    debate_dirs: list[Path] = []
    if args.debate_dirs:
        debate_dirs = [Path(d) for d in args.debate_dirs]
    if args.auto_discover:
        debate_dirs += _find_debate_dirs(Path("outputs"))
    if not debate_dirs:
        # Default: auto-discover
        debate_dirs = _find_debate_dirs(Path("outputs"))
    if not debate_dirs:
        logger.error("No debate dirs found. Use --debate-dir or --auto-discover.")
        sys.exit(1)

    logger.info("Debate dirs: %s", [str(d) for d in debate_dirs])

    out_dir = Path(args.out_dir)
    kg_path = Path(args.kg_path)
    defs_path = Path(args.definitions_path)
    data_dir = Path(args.data_dir)

    breastmnist_root = args.breastmnist_root
    if breastmnist_root is None:
        import os
        breastmnist_root = os.path.expanduser("~/.medmnist")

    all_gold_labels = _load_all_gold_labels(data_dir, breastmnist_root)
    total = sum(len(v) for v in all_gold_labels.values())
    logger.info("Loaded gold labels: train=%d val=%d test=%d (total=%d)",
                len(all_gold_labels["train"]), len(all_gold_labels["val"]),
                len(all_gold_labels["test"]), total)

    index = build_dataset(debate_dirs, out_dir, all_gold_labels, args.k, args.force, kg_path, defs_path)

    # Write index and stats
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    stats = _compute_stats(index)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    logger.info(
        "Dataset complete: %d samples | acc=%.4f | avg_claims=%.1f | avg_triples=%.1f",
        stats.get("num_samples", 0),
        stats.get("accuracy", 0),
        stats.get("avg_claims_per_graph", 0),
        stats.get("avg_triples_per_graph", 0),
    )
    logger.info("Output: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
