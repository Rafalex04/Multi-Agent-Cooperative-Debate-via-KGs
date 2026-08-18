"""Stage 4 gates: round-trip integrity, sign invariant, and the BASE reproduction.

Section 9 of the plan makes these blocking. If BASE does not reproduce the
mal_share AUC that the debate pipeline produced, the graph construction differs
from the pipeline that produced it and every downstream number is
uninterpretable, so this runs before any training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thoth_gnn import CONFIGS, DEFAULTS, ThothGNN, auc, to_hetero  # noqa: E402


def mal_share_auc(graphs, split):
    g = [x for x in graphs if x["split"] == split]
    return auc([x["mal_share"] for x in g], [x["y"] for x in g])


def main(paths):
    ok = True
    for p in paths:
        blob = torch.load(p, weights_only=False)
        graphs, meta = blob["graphs"], blob["meta"]
        name = Path(p).stem
        print(f"\n=== {name} ({len(graphs)} graphs) ===")

        # ---- round trip: corpus -> HeteroData -> batch, counts preserved ----
        cfg = {**DEFAULTS}
        ds = [to_hetero(g, cfg) for g in graphs[:64]]
        from torch_geometric.loader import DataLoader
        b = next(iter(DataLoader(ds, batch_size=64)))
        n_claim = sum(len(g["x_claim"]) for g in graphs[:64])
        n_ag = sum(len(g["agree"]) for g in graphs[:64])
        n_di = sum(len(g["disagree"]) for g in graphs[:64])
        n_ci = sum(len(g["cites"]) for g in graphs[:64])
        checks = [
            ("claim nodes", int(b["claim"].x.size(0)), n_claim),
            ("claim feature dim", int(b["claim"].x.size(1)), 12),
            ("agree edges", int(b["claim", "agree", "claim"].edge_index.size(1)), n_ag),
            ("disagree edges", int(b["claim", "disagree", "claim"].edge_index.size(1)), n_di),
            ("cites edges", int(b["claim", "cites", "triple"].edge_index.size(1)), n_ci),
        ]
        for label, got, want in checks:
            flag = "ok " if got == want else "FAIL"
            ok &= got == want
            print(f"  [{flag}] {label:20s} {got} == {want}")

        # cites weights must be a proper per-target distribution
        w = b["claim", "cites", "triple"].edge_weight
        dst = b["claim", "cites", "triple"].edge_index[1]
        s = torch.zeros(int(b["triple"].x.size(0))).index_add_(0, dst, w)
        bad = int(((s > 1e-6) & ((s - 1).abs() > 1e-3)).sum())
        ok &= bad == 0
        print(f"  [{'ok ' if bad == 0 else 'FAIL'}] cites weights normalise   "
              f"{bad} targets off 1.0")

        # ---- sign invariant: AGREE and DISAGREE never share a weight matrix ----
        m = ThothGNN({**DEFAULTS}, meta["n_relations"])
        wa = m.convs[0].convs[("claim", "agree", "claim")].lin_rel.weight
        wd = m.convs[0].convs[("claim", "disagree", "claim")].lin_rel.weight
        sep = wa.data_ptr() != wd.data_ptr()
        ok &= sep
        print(f"  [{'ok ' if sep else 'FAIL'}] agree/disagree weights are separate tensors")

        # ---- BASE gate ----
        cb = {**DEFAULTS, **CONFIGS["BASE"]}
        mb = ThothGNN(cb, meta["n_relations"])
        assert not mb.gamma.requires_grad and float(mb.gamma) == 0.0
        te = [g for g in graphs if g["split"] == "test"]
        db = [to_hetero(g, cb) for g in te]
        bb = next(iter(DataLoader(db, batch_size=256)))
        mb.eval()
        with torch.no_grad():
            s_model = mb(bb).tolist()
        a_model = auc(s_model, [g["y"] for g in te])
        a_ref = mal_share_auc(graphs, "test")
        same = abs(a_model - a_ref) < 1e-9
        ok &= same
        print(f"  [{'ok ' if same else 'FAIL'}] BASE test AUC {a_model:.6f} == "
              f"mal_share {a_ref:.6f}")
        for sp in ("train", "val", "test"):
            print(f"        mal_share {sp:5s} AUC {mal_share_auc(graphs, sp):.4f}")

    print("\n" + ("ALL GATES PASSED" if ok else "*** GATE FAILURE ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
