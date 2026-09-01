"""Step 9: the single consolidated comparison table.

Every row on the same backbone, data and splits. Baselines are OUR
IMPLEMENTATIONS - neither paper's released code was used. Agent counts differ
between rows and are stated, not hidden.

VLM calls/image are split into PERCEPTION (probe calls the row needs) and
REASONING (debate/verdict turns), because the two baselines and our own methods
spend them very differently and the totals are otherwise misleading.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
RES = _HERE / "results"

# (label, agents, perception calls, reasoning calls, source)
ROWS = [
    ("ResNet-18 supervised (ceiling, not comparable)", "-", 0, 0, "resnet"),
    ("0-parameter KG-signed sum (P3)",                 "-", 16, 0, "static"),
    ("probe-64 flat readout (P2+P3)",                  "-", 32, 0, "static"),
    ("ThothGNN v3 (GNN + real KG)",                    "2", 32, 6, "thoth"),
    ("ThothGNN v3, no graph (A=0)",                    "2", 32, 6, "thoth"),
    ("B2 GraphGeo (ours impl, no KG)",                 "6", 32, 24, "b2"),
    ("B2 GraphGeo + our anchor",                       "6", 32, 24, "b2"),
    ("B2 no-relation (one shared W)",                  "6", 32, 24, "b2"),
    ("B1 Catfish Agent (ours impl)",                   "2+1", 16, 9, "b1"),
    ("B1 no-catfish (= our open-stance protocol)",     "2", 16, 4, "b1"),
    ("B1 catfish sees image (asymmetry removed)",      "2+1", 16, 9, "b1"),
    ("our protocol, 6 agents",                         "6", 0, 24, "ours6"),
    ("self-consistency N=5 (control)",                 "1", 0, 5, "ctl"),
    ("independent ensemble N=2 (control)",             "2", 0, 2, "ctl"),
    ("single draw N=1 (control)",                      "1", 0, 1, "ctl"),
]

STATIC = {
    "0-parameter KG-signed sum (P3)":       (0.7467, 0.6466, 0),
    "probe-64 flat readout (P2+P3)":        (0.8104, 0.7381, 64),
}
THOTH = {
    "ThothGNN v3 (GNN + real KG)":  ("GNN (real KG)", 59),
    "ThothGNN v3, no graph (A=0)":  ("GNN no-graph (A=0)", 59),
}
B2KEY = {
    "B2 GraphGeo (ours impl, no KG)": "B2-full",
    "B2 GraphGeo + our anchor":       "B2-anchored",
    "B2 no-relation (one shared W)":  "B2-no-relation",
}
B1KEY = {
    "B1 Catfish Agent (ours impl)":                "B1-full",
    "B1 no-catfish (= our open-stance protocol)":  "B1-no-catfish",
    "B1 catfish sees image (asymmetry removed)":   "B1-catfish-sees-image",
}
CTLKEY = {
    "self-consistency N=5 (control)":     "self-consistency (N=5)",
    "independent ensemble N=2 (control)": "independent ensemble (N=2)",
    "single draw N=1 (control)":          "single draw (N=1)",
}


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    b1, b2 = jload(RES / "b1.json"), jload(RES / "b2.json")
    ctl, ours6 = jload(RES / "controls.json"), jload(RES / "ours_6agent.json")
    thoth = jload(_HERE.parents[0] / "17_hetgnn/results/thothgnn3.json")
    ext = jload(RES / "external.json") or {}

    print("BreastMNIST test (n=156). Baselines are OUR IMPLEMENTATIONS; no")
    print("released code from either paper was used. Agent counts differ and")
    print("are stated. Ceiling row is not comparable.\n")
    hdr = (f"{'method':46s} {'ag':>4s} {'prm':>6s} {'perc':>5s} {'reas':>5s} "
           f"{'AUC':>15s} {'bAcc':>7s} {'ext AUC':>8s}")
    print(hdr); print("-" * len(hdr))
    out = {}; sd = {}
    for lab, ag, perc, reas, src in ROWS:
        a = b = None; prm = "-"
        if src == "static" and lab in STATIC:
            a, b, prm = STATIC[lab]
        elif src == "thoth" and thoth:
            k, prm = THOTH[lab]
            a, b = thoth[k]["auc"]["test"], thoth[k]["bacc"]
        elif src == "b2" and b2:
            # mean +- sd over the 5 fixed seeds, as the protocol specifies.
            # auc_ens (the AUC of the seed-averaged score) is also stored and
            # runs 0.008-0.016 higher; using it here would flatter B2.
            r = b2.get(B2KEY[lab])
            a, b, prm = (r["auc_mean"], r["bacc"], r["params"]) if r else (None,)*3
            if r:
                sd[lab] = r["auc_sd"]
        elif src == "b1" and b1:
            r = b1.get(B1KEY[lab]);  a, b, prm = (r["auc"], r["bacc"], 0) if r else (None,)*3
        elif src == "ctl" and ctl:
            r = ctl.get(CTLKEY[lab]); a, b, prm = (r["auc"], r["bacc"], 0) if r else (None,)*3
        elif src == "ours6" and ours6:
            r = ours6.get("mal_share over all claims"); a, b, prm = (r["auc"], r["bacc"], 0) if r else (None,)*3
        elif src == "resnet":
            a, b, prm = 0.9442, 0.8672, 11_170_000
        e = ext.get(lab, {}).get("auc")
        out[lab] = {"agents": ag, "params": prm, "perception_calls": perc,
                    "reasoning_calls": reas, "auc": a, "auc_sd": sd.get(lab),
                    "bacc": b, "ext_auc": e}
        pf = (f"{a:.4f}+-{sd[lab]:.3f}" if lab in sd
              else (f"{a:8.4f}" if a is not None else f"{'-':>8s}"))
        bf = f"{b:7.4f}" if b is not None else f"{'-':>7s}"
        ef = f"{e:8.4f}" if e is not None else f"{'pending':>8s}"
        pr = f"{prm:6}" if isinstance(prm, str) else f"{prm:6d}"
        print(f"{lab:46s} {ag:>4s} {pr} {perc:5d} {reas:5d} {pf:>15s} {bf} {ef}")
    (RES / "table.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {RES/'table.json'}")
    print("\nag = agents.  prm = fitted parameters.  perc/reas = VLM calls per image,")
    print("split into perception (probes) and reasoning (debate turns + verdicts).")


if __name__ == "__main__":
    main()
