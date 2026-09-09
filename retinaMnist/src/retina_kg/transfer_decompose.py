"""Which part of the trunk actually carries the transfer?

The retina KG arm is a null (real 0.9043, shuffled 0.8988, no-graph 0.9055), so
Wp and Wn -- the A+ and A- message-passing maps -- may be doing nothing, and the
"trunk transfers" claim could reduce to "W0 transfers". W0 is the per-node feature
map from [probe, mass, net, prior] to hidden; Wp/Wn are the graph channels.

Arms, all with the readout refitted on retina and the trunk frozen:
  full trunk      W0, Wp, Wn, b  from breast
  W0+b only       W0, b from breast;  Wp, Wn random (frozen)
  Wp+Wn only      Wp, Wn from breast; W0, b random (frozen)
  all random      the control already in transfer.py
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P
P.add_experiment_paths("14_kgtensor", "15_kggnn", "17_hetgnn")
import metrics_ordinal as M                                          # noqa: E402
from transfer import (load_retina, load_breast, train_trunk,          # noqa: E402
                      fit_readout_only, score, summarise, TRUNK)

L2, K, SEED, DRAWS = 0.03, 2, 0, 5
R, B = load_retina(), load_breast()
src = train_trunk(B, l2=L2, kdim=K, seed=SEED)
rng = np.random.default_rng(SEED)

def rand_like(k):
    return np.zeros_like(src[k]) if k == "b" else rng.normal(0, 0.3, np.shape(src[k]))

def arm(keep, label, draws=1):
    vals = []
    for _ in range(draws):
        t = {k: (np.array(src[k], copy=True) if k in keep else rand_like(k)) for k in TRUNK}
        Q = fit_readout_only(R, t, l2=L2, kdim=K, seed=SEED)
        s, p, g = score(R, Q)
        vals.append(summarise(R, s, g)["refauc"])
    v = np.array(vals)
    if draws == 1:
        print(f"  {label:34s} refAUC {v[0]:.4f}")
    else:
        print(f"  {label:34s} refAUC {v.mean():.4f} +- {v.std():.4f}")
    return v

print(f"retina ceiling (retina-trained trunk) refAUC 0.9043   [from ordinal_full.json]\n")
full = arm(("W0", "Wp", "Wn", "b"), "full trunk from breast")
w0   = arm(("W0", "b"),             "W0+b from breast, Wp/Wn random", DRAWS)
wpn  = arm(("Wp", "Wn"),            "Wp+Wn from breast, W0/b random", DRAWS)
none = arm((),                      "all random (control)",           DRAWS)

print(f"\n  full - allrandom   {full[0]-none.mean():+.4f}  "
      f"({(full[0]-none.mean())/none.std():+.2f} sd)")
print(f"  W0only - allrandom {w0.mean()-none.mean():+.4f}")
print(f"  WpWnonly-allrandom {wpn.mean()-none.mean():+.4f}")
json.dump({"full": float(full[0]), "w0_b": w0.tolist(), "wp_wn": wpn.tolist(),
           "all_random": none.tolist()},
          open(P.RESULTS / "transfer_decompose.json", "w"), indent=1)
