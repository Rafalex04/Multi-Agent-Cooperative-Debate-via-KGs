"""B2 GraphGeo with the same two ordinal heads as ThothGNN v3.

`experiments/26_baselines/b2_model.py` ends in `nn.Linear(hidden, 1)` and is
trained with `BCEWithLogitsLoss`. That is the same single-scalar bottleneck v3
has, so it takes the same two generalisations, and the A-vs-C comparison stays
like-for-like across the architectures:

  A  one GraphGeo + 4 cut points   (CORAL; +4 params over the binary model)
  C  four independent GraphGeos    (4x the model, nothing shared)

GraphGeo itself is imported unmodified from 26_baselines -- only the head and
the loss live here, so the baseline stays "our implementation of GraphGeo" and
the ordinal change is auditable in one file.

Two things change in the GRAPH for a five-level scale, both in b2_graph_retina:
an "agree" edge joins agents within one grade of each other rather than agents
with an identical binary stance, and the anchor is the mean claim grade rather
than `mal_share`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_experiment_paths                              # noqa: E402
add_experiment_paths("26_baselines")
from b2_model import GraphGeo, collate, drop_edge, REL                # noqa: E402,F401


class CoralGraphGeo(nn.Module):
    """Option A: one GraphGeo trunk, C-1 cut points.

    The trunk keeps its own bias, so unlike the NumPy `ordinal.py` there is one
    redundant degree of freedom between `head` and `theta`. It is harmless under
    weight decay on the head and absent from theta, and it keeps GraphGeo
    byte-identical to the baseline implementation.
    """

    def __init__(self, n_cut, **kw):
        super().__init__()
        self.net = GraphGeo(**kw)
        self.theta = nn.Parameter(torch.zeros(n_cut))
        self.n_cut = n_cut

    def forward(self, *a, **kw):
        s = self.net(*a, **kw)                       # (n,)
        return s[:, None] - self.theta[None, :]      # (n, n_cut) logits

    @staticmethod
    def targets(y, n_cut):
        return (y[:, None] > torch.arange(n_cut, device=y.device)[None, :]).float()


def cum_loss(logits, y, n_cut):
    t = CoralGraphGeo.targets(y, n_cut)
    return nn.functional.binary_cross_entropy_with_logits(logits, t)


def train_eval_coral(tr, te, n_cut, seed, epochs=300, lr=3e-3, wd=5e-4,
                     dropedge=0.2, **kw):
    """One CORAL GraphGeo. Same schedule as b2_train.train_eval."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    B, T = collate(tr), collate(te)
    m = CoralGraphGeo(n_cut, d_img=B["x"].shape[1], **kw)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        e = drop_edge(B["edges"], dropedge, gen)
        z = m(B["x"], e, B["batch"], B["n"], B["slot"], B["anchor"])
        cum_loss(z, B["y"], n_cut).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        z = m(T["x"], T["edges"], T["batch"], T["n"], T["slot"], T["anchor"])
        p = torch.sigmoid(z)
        # theta is a Parameter, so adding it re-attaches grad even inside
        # no_grad on the tensor it touches -- detach before leaving torch.
        s = (z[:, 0] + m.theta[0]).detach()
    return s.numpy(), p.numpy(), (p > 0.5).sum(1).numpy()


def train_eval_stacked(tr, te, n_cut, seed, epochs=300, lr=3e-3, wd=5e-4,
                       dropedge=0.2, **kw):
    """Option C: n_cut independent GraphGeos, each an unmodified binary baseline."""
    import numpy as np
    from b2_train import train_eval

    S = []
    for c in range(n_cut):
        trc = [{**g, "y": float(g["y"] > c)} for g in tr]
        tec = [{**g, "y": float(g["y"] > c)} for g in te]
        S.append(train_eval(trc, tec, seed, epochs=epochs, lr=lr, wd=wd,
                            dropedge=dropedge, **kw))
    S = np.stack(S, axis=1)
    p = 1.0 / (1.0 + np.exp(-np.clip(S, -30, 30)))
    return S, p, (p > 0.5).sum(1)
