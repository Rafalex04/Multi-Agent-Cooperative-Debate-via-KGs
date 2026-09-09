"""Every path this tree uses, resolved once.

The retina work started under experiments/27_retina and moved to its own tree.
Thirty scattered `Path(__file__).parents[N]` expressions made that move a
find-and-replace across a dozen files, so they all live here now: moving the
tree again is a change to TREE alone.

Cross-tree imports are deliberate and stay. `thothgnn3`, `claims`, `gates`,
`run_debate_v5` and `b2_model` are the breast experiments' code, reused rather
than copied so the two datasets run the SAME model -- which is the whole point
of the comparison, and what makes the transfer experiment meaningful.
"""
from __future__ import annotations

import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]          # retinaMnist/
REPO = TREE.parent                                   # repo root
SRC = TREE / "src/retina_kg"

PACK = TREE / "data/retina"                          # ontology + corpora
SOURCE = PACK / "source"                             # the uploaded ICDR json
NPZ = PACK / "images_224"
DEBATES = PACK / "debates_r1"
RESULTS = TREE / "results"
LOGS = TREE / "logs"
ONTOLOGY = TREE / "conf/ontology_retina.yaml"

EXPERIMENTS = REPO / "experiments"
BREAST = REPO / "breastMnist"
BREAST_DEBATES = BREAST / "data/breast/debates_v5q"
BREAST_PROBES = EXPERIMENTS / "15_kggnn/results"


def add_experiment_paths(*names: str) -> None:
    """Put this tree and the named experiment dirs on sys.path, in order."""
    for p in (str(SRC), *[str(EXPERIMENTS / n) for n in names]):
        if p not in sys.path:
            sys.path.insert(0, p)
