"""Compare pre-fix (28px) and post-fix (224px) debate graphs on the same 12 samples.

The 12 probe samples are a subset of ablation_indices.json, so every one of them
already has a graph built from the low-resolution debates in
data/breast/dataset_full/test/graphs/. This script puts the two side by side.

Primary metric is claim-text grounding, NOT label separation: under the current
adversarial prompts a claim's label is fixed by which expert wrote it
(corr(frac_claims_by_B, frac_mal_claims) = 0.96), so frac_mal_claims cannot move
with image quality no matter how good the pixels are. Read the diversity and
cross-sample-similarity rows first.

Usage:
  python scripts/compare_probe12.py \
      --before data/breast/dataset_full/test/graphs \
      --after  data/breast/dataset_probe12/test/graphs
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

PROBE_IDS = ["007", "052", "114", "124", "024", "033", "150", "050", "107", "108", "038", "115"]


def _norm(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — for duplicate detection."""
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower().strip())
    return re.sub(r"\s+", " ", t)


def _toks(text: str) -> set[str]:
    return set(_norm(text).split())


def _load(graphs_dir: Path, sample_ids: list[str]) -> list[dict]:
    graphs = []
    for sid in sample_ids:
        p = graphs_dir / f"sample_{sid}.json"
        if p.exists():
            graphs.append(json.loads(p.read_text()))
    return graphs


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def compute_metrics(graphs: list[dict]) -> dict[str, float]:
    """Return grounding/diversity metrics for a set of sample graphs."""
    if not graphs:
        return {}

    all_claims = [c for g in graphs for c in g["nodes"]["claims"]]
    norm_claims = [_norm(c["text"]) for c in all_claims]

    # Template collapse: how similar are debates for DIFFERENT images?
    docs = [_toks(" ".join(c["text"] for c in g["nodes"]["claims"])) for g in graphs]
    cross = [
        _jaccard(docs[i], docs[j])
        for i in range(len(docs))
        for j in range(i + 1, len(docs))
    ]

    # Self-recycling: does an expert restate itself between rounds?
    recycle = []
    for g in graphs:
        by_expert: dict[str, dict[int, set[str]]] = collections.defaultdict(dict)
        for c in g["nodes"]["claims"]:
            r = c["round_idx"]
            by_expert[c["expert_id"]][r] = by_expert[c["expert_id"]].get(r, set()) | _toks(c["text"])
        for rounds in by_expert.values():
            ordered = sorted(rounds)
            for a, b in zip(ordered, ordered[1:]):
                recycle.append(_jaccard(rounds[a], rounds[b]))

    # Within-sample duplicate claims
    dup_rates = []
    for g in graphs:
        texts = [_norm(c["text"]) for c in g["nodes"]["claims"]]
        if texts:
            dup_rates.append(1 - len(set(texts)) / len(texts))

    # Label separation (expected to stay flat while stances are locked)
    frac_mal: dict[str, list[float]] = {"BENIGN": [], "MALIGNANT": []}
    for g in graphs:
        labels = collections.Counter(c["label"] for c in g["nodes"]["claims"] if c["label"])
        total = sum(labels.values())
        if total:
            frac_mal[g["gold_label"]].append(labels["MALIGNANT"] / total)

    triples = {t["text"] for g in graphs for t in g["nodes"]["triples"]}

    return {
        "n_samples": len(graphs),
        "unique_claim_ratio": len(set(norm_claims)) / len(norm_claims) if norm_claims else 0,
        "cross_sample_jaccard": statistics.mean(cross) if cross else 0,
        "self_recycle_jaccard": statistics.mean(recycle) if recycle else 0,
        "within_dup_rate": statistics.mean(dup_rates) if dup_rates else 0,
        "distinct_kg_triples": len(triples),
        "avg_claims": statistics.mean(len(g["nodes"]["claims"]) for g in graphs),
        "accuracy": sum(g["correct"] for g in graphs) / len(graphs),
        "frac_mal_BENIGN": statistics.mean(frac_mal["BENIGN"]) if frac_mal["BENIGN"] else float("nan"),
        "frac_mal_MALIGNANT": statistics.mean(frac_mal["MALIGNANT"]) if frac_mal["MALIGNANT"] else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pre-fix vs post-fix probe graphs")
    parser.add_argument("--before", default="data/breast/dataset_full/test/graphs")
    parser.add_argument("--after", default="data/breast/dataset_probe12/test/graphs")
    args = parser.parse_args()

    before = _load(Path(args.before), PROBE_IDS)
    after = _load(Path(args.after), PROBE_IDS)

    if not before:
        raise SystemExit(f"No pre-fix graphs found in {args.before}")
    if not after:
        raise SystemExit(f"No post-fix graphs found in {args.after} — run the probe first")

    mb, ma = compute_metrics(before), compute_metrics(after)

    # (label, key, direction) — direction is what "better" looks like
    rows = [
        ("samples compared",        "n_samples",            ""),
        ("unique claim ratio",      "unique_claim_ratio",   "higher"),
        ("cross-sample jaccard",    "cross_sample_jaccard", "lower"),
        ("self-recycle jaccard",    "self_recycle_jaccard", "lower"),
        ("within-sample dup rate",  "within_dup_rate",      "lower"),
        ("distinct KG triples",     "distinct_kg_triples",  "higher"),
        ("avg claims/graph",        "avg_claims",           ""),
        ("accuracy",                "accuracy",             "higher"),
        ("frac_mal | gold BENIGN",  "frac_mal_BENIGN",      ""),
        ("frac_mal | gold MALIG",   "frac_mal_MALIGNANT",   ""),
    ]

    print(f"\n{'metric':<26} {'BEFORE (28px)':>14} {'AFTER (224px)':>14} {'delta':>10}  better")
    print("-" * 78)
    for label, key, direction in rows:
        b, a = mb.get(key, float("nan")), ma.get(key, float("nan"))
        print(f"{label:<26} {b:>14.4f} {a:>14.4f} {a - b:>+10.4f}  {direction}")

    sep_b = abs(mb["frac_mal_MALIGNANT"] - mb["frac_mal_BENIGN"])
    sep_a = abs(ma["frac_mal_MALIGNANT"] - ma["frac_mal_BENIGN"])
    print("-" * 78)
    print(f"{'label separation |diff|':<26} {sep_b:>14.4f} {sep_a:>14.4f} {sep_a - sep_b:>+10.4f}  higher")
    print(
        "\nNote: label separation is expected to stay near zero while the adversarial\n"
        "prompts forbid stance changes. Judge this run on the diversity rows above.\n"
    )


if __name__ == "__main__":
    main()
