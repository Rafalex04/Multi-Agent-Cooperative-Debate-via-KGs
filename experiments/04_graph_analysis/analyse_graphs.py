"""Measure whether a debate-graph dataset carries enough signal to train a GNN.

A GNN can only beat the debate verdict if the graph structure encodes something
the verdict does not. This script quantifies that in four parts:

  1. SEPARABILITY  Do graph-level features differ between malignant and benign
                   graphs? Reported as single-feature AUC, so 0.5 means the
                   feature is pure noise.
  2. READOUT       Can a trivial readout (claim-label majority) beat the debate
                   verdict? If it cannot, message passing has little to build on.
  3. DIVERSITY     Are the claims varied, or do the agents repeat one template?
                   Repetition means near-identical graphs and no learnable
                   structure.
  4. DEGENERACY    How many graphs are structurally identical or near-identical?
                   Duplicate graphs with opposite labels cap achievable accuracy.

Usage:
  python analyse_graphs.py --dataset ../../breastMnist/data/breast/dataset_full
"""
from __future__ import annotations

import argparse, json, math, re
from collections import Counter, defaultdict
from pathlib import Path


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability a random positive outranks a random negative (ties = 0.5)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def load_split(root: Path, split: str) -> list[dict]:
    d = root / split / "graphs"
    if not d.is_dir():
        return []
    return [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]


def graph_features(g: dict) -> dict:
    """Structural and content features computable without the gold label."""
    claims = g["nodes"]["claims"]
    triples = g["nodes"]["triples"]
    cc = g["edges"]["claim_claim"]

    n_claims = len(claims)
    mal = sum(1 for c in claims if c.get("label") == "MALIGNANT")
    ben = sum(1 for c in claims if c.get("label") == "BENIGN")
    labelled = mal + ben

    agree    = sum(1 for e in cc if e.get("type") == "AGREE")
    disagree = sum(1 for e in cc if e.get("type") == "DISAGREE")

    # Per-expert malignant share: captures whether the two agents diverged.
    by_expert: dict[str, list[str]] = defaultdict(list)
    for c in claims:
        by_expert[c.get("expert_id", "?")].append(c.get("label", "?"))
    shares = []
    for _, labs in sorted(by_expert.items()):
        n = sum(1 for l in labs if l in ("MALIGNANT", "BENIGN"))
        if n:
            shares.append(sum(1 for l in labs if l == "MALIGNANT") / n)
    expert_gap = abs(shares[0] - shares[1]) if len(shares) == 2 else 0.0

    # Malignant share in the final round only: the debate's settled position.
    rounds = [c.get("round_idx", 0) for c in claims]
    last_r = max(rounds) if rounds else 0
    final = [c for c in claims if c.get("round_idx", 0) == last_r]
    fin_lab = [c["label"] for c in final if c.get("label") in ("MALIGNANT", "BENIGN")]
    final_share = (sum(1 for l in fin_lab if l == "MALIGNANT") / len(fin_lab)
                   if fin_lab else 0.5)

    feats = {
        "n_claims":       n_claims,
        "n_triples":      len(triples),
        "mal_share":      mal / labelled if labelled else 0.5,
        "final_share":    final_share,
        "expert_gap":     expert_gap,
        "n_cc_edges":     len(cc),
        "agree":          agree,
        "disagree":       disagree,
        "disagree_ratio": disagree / len(cc) if cc else 0.0,
        "n_ct_edges":     len(g["edges"]["claim_triple"]),
        "n_tt_edges":     len(g["edges"]["triple_triple"]),
        "rounds_used":    g.get("rounds_used", 0),
    }

    # v2 graphs carry measured evidence on each claim node. These are the
    # attributes a GNN would actually message-pass over, so they matter more
    # than the structural counts above.
    if g.get("evidence_score") is not None:
        feats["evidence_score"] = g["evidence_score"]
    pys = [c["p_yes"] for c in claims if c.get("p_yes") is not None]
    if pys:
        feats["mean_p_yes"] = sum(pys) / len(pys)
        # Claim-level readout: probe confidence signed by the KG stance of the
        # feature the claim cites.
        signed = [c["p_yes"] * c.get("stance_weight", 0.0)
                  for c in claims if c.get("p_yes") is not None]
        feats["mean_signed_p"] = sum(signed) / len(signed)
        feats["frac_cited"] = sum(
            1 for c in claims if c.get("cited_feature")) / n_claims if n_claims else 0.0
    return feats


_WORD = re.compile(r"[a-z]+")


def analyse(graphs: list[dict], name: str) -> dict:
    if not graphs:
        return {}
    print(f"\n{'='*70}\n{name}  (n={len(graphs)})\n{'='*70}")

    gold = [g["gold_label"] for g in graphs]
    n_mal = sum(1 for x in gold if x == "MALIGNANT")
    print(f"Class balance: {n_mal} malignant / {len(gold)-n_mal} benign "
          f"({n_mal/len(gold):.1%} positive)")

    # --- Debate verdict as the baseline a GNN must beat ---
    verdicts = [g.get("verdict") for g in graphs]
    correct  = sum(1 for g in graphs if g.get("correct"))
    v_tp = sum(1 for g, v in zip(graphs, verdicts)
               if g["gold_label"] == "MALIGNANT" and v == "MALIGNANT")
    v_fp = sum(1 for g, v in zip(graphs, verdicts)
               if g["gold_label"] == "BENIGN" and v == "MALIGNANT")
    v_fn = n_mal - v_tp
    v_tn = (len(gold) - n_mal) - v_fp
    sens = v_tp / n_mal if n_mal else 0
    spec = v_tn / (len(gold) - n_mal) if len(gold) - n_mal else 0
    print(f"\nDebate verdict: acc={correct/len(graphs):.4f}  "
          f"balanced_acc={(sens+spec)/2:.4f}")
    print(f"  TP={v_tp} FP={v_fp} TN={v_tn} FN={v_fn}  "
          f"sens={sens:.3f} spec={spec:.3f}")
    vc = Counter(verdicts)
    print(f"  verdict distribution: {dict(vc)}")

    # --- 1. Single-feature separability ---
    print(f"\n--- 1. SEPARABILITY (single-feature AUC; 0.5 = noise) ---")
    feats = [graph_features(g) for g in graphs]
    # Evidence keys are absent in v1 graphs, so take the union and skip gaps.
    keys = list(dict.fromkeys(k for f in feats for k in f))
    sep = {}
    for k in keys:
        pos = [f[k] for f, gl in zip(feats, gold) if gl == "MALIGNANT" and k in f]
        neg = [f[k] for f, gl in zip(feats, gold) if gl == "BENIGN" and k in f]
        a = auc(pos, neg)
        if a is None:
            continue
        sep[k] = a
        mp = sum(pos) / len(pos)
        mn = sum(neg) / len(neg)
        flag = "  <-- signal" if abs(a - 0.5) >= 0.08 else ""
        print(f"  {k:16s} AUC={a:.4f}  mean(mal)={mp:8.3f} mean(ben)={mn:8.3f}{flag}")

    # --- 2. Trivial readout vs verdict ---
    print(f"\n--- 2. READOUT (can a trivial graph readout beat the verdict?) ---")
    for k in ("mal_share", "final_share", "mean_signed_p", "evidence_score"):
        if not any(k in f for f in feats):
            continue
        scores = [f.get(k, 0.5) for f in feats]
        pos = [s for s, gl in zip(scores, gold) if gl == "MALIGNANT"]
        neg = [s for s, gl in zip(scores, gold) if gl == "BENIGN"]
        a = auc(pos, neg)
        # Best achievable balanced accuracy by sweeping the threshold.
        best_ba, best_th = 0.0, 0.0
        for th in sorted(set(scores)):
            pred = [s >= th for s in scores]
            tp = sum(1 for p, gl in zip(pred, gold) if p and gl == "MALIGNANT")
            tn = sum(1 for p, gl in zip(pred, gold) if not p and gl == "BENIGN")
            ba = 0.5 * (tp / n_mal + tn / (len(gold) - n_mal))
            if ba > best_ba:
                best_ba, best_th = ba, th
        print(f"  {k:12s} AUC={a:.4f}  best_balanced_acc={best_ba:.4f} @ th={best_th:.3f}")

    # --- 3. Claim diversity ---
    print(f"\n--- 3. DIVERSITY (are the arguments varied?) ---")
    all_texts, per_graph_uniq = [], []
    for g in graphs:
        txts = [c["text"].strip().lower() for c in g["nodes"]["claims"]]
        all_texts += txts
        per_graph_uniq.append(len(set(txts)) / len(txts) if txts else 0)
    uniq_global = len(set(all_texts)) / len(all_texts) if all_texts else 0
    print(f"  claims total={len(all_texts)}  globally unique={uniq_global:.3f}")
    print(f"  mean within-graph uniqueness={sum(per_graph_uniq)/len(per_graph_uniq):.3f}")
    vocab = Counter(w for t in all_texts for w in _WORD.findall(t))
    print(f"  vocabulary={len(vocab)} distinct words")
    print(f"  10 most repeated claims:")
    for txt, cnt in Counter(all_texts).most_common(10):
        print(f"    {cnt:4d}x  {txt[:88]}")

    # --- 4. Degeneracy ---
    print(f"\n--- 4. DEGENERACY (identical graphs with conflicting labels) ---")
    sig_map: dict[tuple, list[str]] = defaultdict(list)
    for g, f in zip(graphs, feats):
        sig = (tuple(sorted(c["text"].strip().lower() for c in g["nodes"]["claims"])),)
        sig_map[sig].append(g["gold_label"])
    dup_groups = {s: l for s, l in sig_map.items() if len(l) > 1}
    conflicting = sum(len(l) for l in dup_groups.values()
                      if len(set(l)) > 1)
    print(f"  distinct claim-sets={len(sig_map)} of {len(graphs)} graphs")
    print(f"  graphs in duplicate groups={sum(len(l) for l in dup_groups.values())}")
    print(f"  graphs in label-conflicting duplicate groups={conflicting}")

    # Feature-vector collisions: identical structure, different label.
    fsig: dict[tuple, list[str]] = defaultdict(list)
    for f, gl in zip(feats, gold):
        fsig[tuple(round(f[k], 4) for k in keys if k in f)].append(gl)
    fconf = sum(len(l) for l in fsig.values() if len(set(l)) > 1)
    print(f"  distinct structural feature vectors={len(fsig)} of {len(graphs)}")
    print(f"  graphs with a structural twin of the opposite label={fconf} "
          f"({fconf/len(graphs):.1%})")

    return {"n": len(graphs), "verdict_acc": correct / len(graphs),
            "separability": sep,
            "uniq_global": uniq_global,
            "struct_conflict_rate": fconf / len(graphs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--splits", default="train,val,test")
    args = p.parse_args()

    root = Path(args.dataset)
    summary = {}
    all_graphs = []
    for split in args.splits.split(","):
        gs = load_split(root, split)
        if gs:
            summary[split] = analyse(gs, f"SPLIT: {split}")
            all_graphs += gs

    if len(summary) > 1:
        summary["ALL"] = analyse(all_graphs, "ALL SPLITS COMBINED")

    print(f"\n{'='*70}\nVERDICT ON GNN VIABILITY\n{'='*70}")
    a = summary.get("ALL") or next(iter(summary.values()))
    best = max(a["separability"].items(), key=lambda kv: abs(kv[1] - 0.5))
    print(f"Strongest single feature : {best[0]} (AUC={best[1]:.4f})")
    print(f"Debate verdict accuracy  : {a['verdict_acc']:.4f}")
    print(f"Claim uniqueness         : {a['uniq_global']:.3f}")
    print(f"Structural label conflict: {a['struct_conflict_rate']:.1%}")
    print()
    if abs(best[1] - 0.5) < 0.06:
        print("=> WEAK. No graph feature separates the classes. A GNN has nothing")
        print("   to learn beyond the class prior. Regenerate the dataset.")
    elif a["struct_conflict_rate"] > 0.30:
        print("=> WEAK. Too many graphs are structurally identical but carry")
        print("   opposite labels; this caps achievable accuracy. Regenerate.")
    else:
        print("=> USABLE. At least one feature separates the classes and graphs")
        print("   are mostly distinguishable. A GNN is worth training.")


if __name__ == "__main__":
    main()
