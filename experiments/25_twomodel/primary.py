"""Primary endpoint, exactly as pre-registered on 2026-08-30.

Difference of differences: does the debate channel contribute MORE when the two
debaters are different models than when they are the same one?

The probe channel is identical in both arms by construction, so "probes only" is
the same vector in both and the DoD reduces to a paired comparison of the two
"probes + debate" readouts on the same samples. Bar: P >= 0.95.
"""
import json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "16_external"))
from analyse_e1 import auc, paired_bootstrap          # noqa: E402

hom = json.loads((_HERE / "results/merged_v5_hom.json").read_text())
het = json.loads((_HERE / "results/merged_v5_het.json").read_text())

# align on the sample keys present in both arms
kh, ke = hom["keys"], het["keys"]
common = [k for k in kh if k in set(ke)]
ih = {k: i for i, k in enumerate(kh)}
ie = {k: i for i, k in enumerate(ke)}
sel_h = np.array([ih[k] for k in common])
sel_e = np.array([ie[k] for k in common])
y = np.array(hom["y"])[sel_h]
assert (y == np.array(het["y"])[sel_e]).all(), "label misalignment between arms"

P_h = np.array(hom["oof"]["probes only"])[sel_h]
P_e = np.array(het["oof"]["probes only"])[sel_e]
D_h = np.array(hom["oof"]["probes + debate"])[sel_h]
D_e = np.array(het["oof"]["probes + debate"])[sel_e]

a_ph, a_pe = auc(y, P_h), auc(y, P_e)
a_dh, a_de = auc(y, D_h), auc(y, D_e)
print(f"n = {len(common)} images, malignant {int(y.sum())}\n")
print(f"{'arm':28s} {'probes':>9s} {'+debate':>9s} {'contribution':>14s}")
print(f"{'HOM  qwen x qwen':28s} {a_ph:9.4f} {a_dh:9.4f} {a_dh-a_ph:+14.4f}")
print(f"{'HET  qwen x minicpm':28s} {a_pe:9.4f} {a_de:9.4f} {a_de-a_pe:+14.4f}")

dod = (a_de - a_pe) - (a_dh - a_ph)
p = paired_bootstrap(y, D_e, D_h, n=4000)
print(f"\nPRIMARY  difference of differences = {dod:+.4f}")
print(f"         P(HET debate-contribution > HOM) = {p:.3f}")
print(f"\nDECISION (bar P >= 0.95): {'CLEARS' if p >= 0.95 else 'DOES NOT CLEAR'} the bar")
print(f"\nBoth arms remain NEGATIVE: the debate channel costs "
      f"{-(a_dh-a_ph):.4f} (HOM) and {-(a_de-a_pe):.4f} (HET) of AUC.")
json.dump({"n": len(common), "hom_probes": a_ph, "hom_debate": a_dh,
           "het_probes": a_pe, "het_debate": a_de,
           "hom_contribution": a_dh - a_ph, "het_contribution": a_de - a_pe,
           "dod": dod, "P_het_better": float(p), "bar": 0.95,
           "clears": bool(p >= 0.95)},
          open(_HERE / "results/primary.json", "w"), indent=1)
