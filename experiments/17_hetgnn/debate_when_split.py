"""Debate only the samples the two agents already disagree on.

The proposal is about COST and GRAPH SIZE, not only accuracy: if the two experts
open with the same verdict, the debate that follows is two agents agreeing with
each other at length, and every claim and edge it produces is graph the model has
to carry. Skipping those samples would shrink the debate graph to the cases that
are actually contested.

This is answerable on the existing corpus without regenerating anything, because
every claim records its expert and its round. Each agent's OPENING verdict is the
majority label of its round-0 claims, which is the state of the world before any
argument has happened -- exactly the quantity the gate would be applied to.

Four questions, in the order they decide whether the idea is worth building:

  1. How much would it actually save? Fraction of samples where the openings
     split, and the claims and edges that live on each side.
  2. Is skipping SAFE? When the two agents open in agreement, how often are they
     right? If the agreed cases are near-perfect, skipping them costs nothing.
  3. Does the debate EARN its cost on the contested cases? On the split subset,
     does anything downstream of the opening -- the later rounds, the attack
     edges -- resolve them better than the opening did?
  4. Does the hybrid beat the alternatives end to end?

Complete BUS-BRA corpus: 1875 debates, 1064 cases.
"""
from __future__ import annotations

import glob, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _d in ("14_kgtensor", "15_kggnn", "16_external"):
    sys.path.insert(0, str(_HERE.parents[1] / _d))
from analyse_e1 import auc, bacc, load_probes, paired_bootstrap          # noqa: E402
from claims import kg_findings                                           # noqa: E402
from gates import fit_rank                                               # noqa: E402
from powered_gnn import case_folds                                       # noqa: E402

_DEB = _HERE.parents[2] / "breastMnist/data/breast/debates_busbra/all"
_EXT = _HERE.parents[1] / "16_external/results"


def verdict(cl):
    """Majority MALIGNANT-ness of a claim list; None when it has no claims."""
    if not cl:
        return None
    m = sum(1 for c in cl if c.get("label") == "MALIGNANT")
    return m / len(cl)


def main():
    findings = kg_findings()
    names = [f for f, _ in findings]
    prior = np.array([s for _, s in findings], float)

    rec = {}
    for f in sorted(glob.glob(str(_DEB / "*.json"))):
        d = json.loads(Path(f).read_text())
        cl = d.get("claims") or []
        if not cl:
            continue
        opens = {}
        for a in ("agent_1", "agent_2"):
            opens[a] = verdict([c for c in cl
                                if c.get("expert_id") == a and int(c.get("round_idx", 0)) == 0])
        if opens["agent_1"] is None or opens["agent_2"] is None:
            continue
        last = max(int(c.get("round_idx", 0)) for c in cl)
        rec[int(d["sample_id"])] = {
            "y": 1.0 if d["gold_label"] == "MALIGNANT" else 0.0,
            "o1": opens["agent_1"], "o2": opens["agent_2"],
            "open_all": verdict([c for c in cl if int(c.get("round_idx", 0)) == 0]),
            "final": verdict([c for c in cl if int(c.get("round_idx", 0)) == last]),
            "all": verdict(cl),
            "n_claims": len(cl),
            "n_edges": sum(len(c.get("addressed_ids") or []) for c in cl),
            "n_after": sum(1 for c in cl if int(c.get("round_idx", 0)) > 0),
        }

    probes = load_probes(names, ("busbra_p2", "busbra_p3"), _EXT)
    common = sorted(set(rec) & set(probes))
    z = np.load(_HERE.parents[2] / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    cases = z["cases"][common]
    y = np.array([rec[i]["y"] for i in common])
    n = len(common)

    o1 = np.array([rec[i]["o1"] for i in common])
    o2 = np.array([rec[i]["o2"] for i in common])
    # an agent "votes" malignant when the majority of its opening claims do
    v1, v2 = (o1 > 0.5), (o2 > 0.5)
    split = v1 != v2
    print(f"complete corpus: {n} images / {len(np.unique(cases))} cases")

    # ---- 1. what would be saved ------------------------------------------
    nc = np.array([rec[i]["n_claims"] for i in common])
    ne = np.array([rec[i]["n_edges"] for i in common])
    na = np.array([rec[i]["n_after"] for i in common])
    print("\n=== 1. what gating on the opening split would save ===")
    print(f"  agents open SPLIT      {split.sum():5d} / {n}  ({100*split.mean():.1f}%)")
    print(f"  agents open AGREED     {(~split).sum():5d} / {n}  ({100*(~split).mean():.1f}%)")
    print(f"  post-opening claims on agreed samples : {na[~split].sum():6d} "
          f"({100*na[~split].sum()/na.sum():.1f}% of all argument produced)")
    print(f"  attack/endorse edges on agreed samples: {ne[~split].sum():6d} "
          f"({100*ne[~split].sum()/ne.sum():.1f}% of all edges)")
    print(f"  mean claims  split {nc[split].mean():.1f}  vs agreed {nc[~split].mean():.1f}")
    print(f"  mean edges   split {ne[split].mean():.1f}  vs agreed {ne[~split].mean():.1f}")

    # ---- 2. is skipping safe? --------------------------------------------
    agreed_call = v1[~split].astype(float)
    acc = (agreed_call == y[~split]).mean()
    base_rate = max(y[~split].mean(), 1 - y[~split].mean())
    print("\n=== 2. when the two agents open in agreement, are they right? ===")
    print(f"  accuracy of the agreed opening call   {acc:.4f}  "
          f"(majority-class baseline on that subset {base_rate:.4f})")
    print(f"  accuracy on the SPLIT subset, agent_1 {(v1[split]==y[split]).mean():.4f}"
          f"   agent_2 {(v2[split]==y[split]).mean():.4f}")
    print(f"  malignant prevalence  agreed {y[~split].mean():.3f}   split {y[split].mean():.3f}")

    # ---- 3. does arguing change anything on the contested cases? ---------
    fin = np.array([rec[i]["final"] for i in common])
    opn = np.array([rec[i]["open_all"] for i in common])
    print("\n=== 3. on the contested cases, does the argument resolve them? ===")
    for lab, m in (("SPLIT", split), ("AGREED", ~split)):
        if m.sum() < 20:
            continue
        print(f"  {lab:6s} n={m.sum():4d}   opening-pool AUC {auc(y[m], opn[m]):.4f}"
              f"   final-round AUC {auc(y[m], fin[m]):.4f}"
              f"   all-claims AUC {auc(y[m], np.array([rec[i]['all'] for i in common])[m]):.4f}")

    # ---- 4. end to end ----------------------------------------------------
    P = np.array([probes[i] for i in common])
    F = len(names)
    Zp = ((P.reshape(n, -1) - P.reshape(n, -1).mean(0))
          / np.where(P.reshape(n, -1).std(0) < 1e-9, 1, P.reshape(n, -1).std(0)))
    dsc = np.array([rec[i]["all"] for i in common])
    folds = case_folds(cases, k=5, seed=0)
    u = np.unique(cases)
    yc = np.array([y[cases == c][0] for c in u])
    to_case = lambda v: np.array([v[cases == c].mean() for c in u])

    def oof(cols):
        o = np.zeros(n)
        for te in folds:
            w, b = fit_rank(cols[~te], y[~te], 0.3)
            o[te] = cols[te] @ w + b
        return o

    base = np.hstack([Zp, np.tile(prior, (n, 1))])
    full = np.hstack([base, dsc[:, None]])
    gated = dsc.copy(); gated[~split] = opn[~split]      # skip the argument when they agree
    hyb = np.hstack([base, gated[:, None]])
    print("\n=== 4. end to end, 5-fold on cases ===")
    s_b, s_f, s_h = to_case(oof(base)), to_case(oof(full)), to_case(oof(hyb))
    print(f"  probes only                                  {auc(yc, s_b):.4f}")
    print(f"  + debate score, every sample debated         {auc(yc, s_f):.4f}"
          f"   P vs probes {paired_bootstrap(yc, s_f, s_b):.3f}")
    print(f"  + debate score, ONLY where the agents split  {auc(yc, s_h):.4f}"
          f"   P vs probes {paired_bootstrap(yc, s_h, s_b):.3f}"
          f"   P vs full {paired_bootstrap(yc, s_h, s_f):.3f}")

    out = {"n": n, "frac_split": float(split.mean()),
           "claims_saved_frac": float(na[~split].sum() / na.sum()),
           "edges_saved_frac": float(ne[~split].sum() / ne.sum()),
           "agreed_open_accuracy": float(acc), "agreed_base_rate": float(base_rate),
           "auc": {"probes": auc(yc, s_b), "debate_all": auc(yc, s_f),
                   "debate_split_only": auc(yc, s_h)}}
    (_HERE.parent / "results/debate_when_split.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\nwrote results/debate_when_split.json")


if __name__ == "__main__":
    main()
