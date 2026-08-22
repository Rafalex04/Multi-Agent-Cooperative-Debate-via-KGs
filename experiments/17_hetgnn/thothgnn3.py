"""ThothGNN v3 -- message passing on the KG, fed by the debate through citations.

Built to what the diagnostics actually support rather than to what would be
tidy. Three measured facts set the architecture:

  1. The claim->claim attack graph carries no information about correctness
     (AUC(credibility -> correct) = 0.5017). So there is no propagation over
     attacks here. Including it would be decoration.
  2. The claim->finding citation edges DO carry signal -- argument mass read
     through the KG prior reaches 0.6983 on test by itself. So the debate enters
     on the finding nodes, along the citation edges it actually emitted.
  3. The KG's own structure carries signal (ablation 0.7128 vs 0.4740). So the
     graph the network passes messages on is the knowledge graph.

The merge is therefore claim --cites--> finding --kg--> finding, and the GNN runs
on that. Node i is a BI-RADS finding carrying what was SEEN (two probe phrasings),
what was ARGUED about it (mass and net stance, pushed along citation edges), and
what the ontology SAYS it means (the stance prior).

    H = relu( X W0 + A+ X Wp + A- X Wn + b )
    s = sum_f u_f . H_f + c

Design constraints inherited from earlier failures in this project, all of which
cost a run to learn:

  * No multiplicative gate on a zero-initialised linear. gamma*(w.x+b) with both
    at zero has zero gradient in both and never leaves the dead point.
  * A KG prior is a PENALTY CENTRE, never an initialisation: under a convex
    L2-penalised objective the optimum is unique, so an initialisation cannot
    survive to it and the hypothesis would be untestable.
  * Sum aggregation, not mean. Mean cannot distinguish "one finding argued
    heavily" from "many findings argued lightly", and that distinction is the
    whole content of the citation channel.
  * Every number is reported against a shuffled-KG control with the same degree
    sequence, because a dense 16-node graph helps a little no matter what it says.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
sys.path.insert(0, str(_HERE.parents[1] / "15_kggnn"))
from claims import auc, bacc                                          # noqa: E402
from hetgraph import SPLITS, build, kg_operators                      # noqa: E402

_PROBES = _HERE.parents[1] / "15_kggnn/results"
TAGS = ("probeneg", "probep3")            # P2 + P3, the registered pair


# ------------------------------------------------------------------ features --
def probe_block(names):
    """(split, index) -> [p_yes for each tag] per finding."""
    out = {}
    for ti, t in enumerate(TAGS):
        for f in sorted(Path(_PROBES).glob(f"{t}_*.jsonl")):
            split = f.name.split("_")[1]
            for ln in f.read_text(errors="replace").splitlines():
                if not ln.strip():
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                k = (split, int(r["index"]))
                a = out.setdefault(k, np.full((len(names), len(TAGS)), 0.5))
                for j, n in enumerate(names):
                    v = (r["p_yes"] or {}).get(n)
                    if v is not None:
                        a[j, ti] = float(v)
    return out


def node_tensor(g, names, prior, probes, split):
    """n x F x 5 : [probe_P2, probe_P3, argument mass, net stance, prior]."""
    F = len(names)
    n = len(g["y"])
    X = np.zeros((n, F, 5))
    mass = np.einsum("nc,ncf->nf", g["mask"], g["Acf"])
    net = np.einsum("nc,ncf->nf", g["mask"] * g["Xc"][:, :, 0], g["Acf"])
    for i, sid in enumerate(g["ids"]):
        p = probes.get((split, int(sid)))
        if p is not None:
            X[i, :, 0:2] = p
        else:
            X[i, :, 0:2] = 0.5
    X[:, :, 2] = mass
    X[:, :, 3] = net
    X[:, :, 4] = prior[None, :]
    return X


# ----------------------------------------------------------------- the model --
def forward(P, X, Ap, An):
    """H = relu(X W0 + A+ X Wp + A- X Wn + b) ; s = sum_f H_f . u + c"""
    M = X @ P["W0"] + np.einsum("fg,ngd->nfd", Ap, X) @ P["Wp"] \
        + np.einsum("fg,ngd->nfd", An, X) @ P["Wn"] + P["b"]
    H = np.maximum(M, 0.0)
    return H, M, (H * P["u"]).sum(axis=(1, 2)) + P["c"]


def grads(P, X, Ap, An, y, l2, w0_centre):
    H, M, s = forward(P, X, Ap, An)
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    e = (p - y) / len(y)
    dH = e[:, None, None] * P["u"]
    dM = dH * (M > 0)
    ApX = np.einsum("fg,ngd->nfd", Ap, X)
    AnX = np.einsum("fg,ngd->nfd", An, X)
    G = {
        "W0": np.einsum("nfd,nfk->dk", X, dM) + l2 * P["W0"],
        "Wp": np.einsum("nfd,nfk->dk", ApX, dM) + l2 * P["Wp"],
        "Wn": np.einsum("nfd,nfk->dk", AnX, dM) + l2 * P["Wn"],
        "b": dM.sum(axis=(0, 1)) + l2 * P["b"],
        "u": np.einsum("n,nfk->fk", e, H) + l2 * (P["u"] - w0_centre),
        "c": float(e.sum()),
    }
    return G, s


def init(F, d, k, rng, prior):
    """u is centred on the KG prior as a PENALTY target, not seeded with it."""
    P = {"W0": rng.normal(0, 0.3, (d, k)), "Wp": rng.normal(0, 0.3, (d, k)),
         "Wn": rng.normal(0, 0.3, (d, k)), "b": np.zeros(k),
         "u": rng.normal(0, 0.3, (F, k)), "c": 0.0}
    return P


def fit(X, y, Ap, An, l2=0.1, kdim=4, epochs=800, lr=0.15, seed=0, prior_centre=None):
    F, d = X.shape[1], X.shape[2]
    rng = np.random.default_rng(seed)
    P = init(F, d, kdim, rng, prior_centre)
    centre = np.zeros((F, kdim))
    if prior_centre is not None:
        centre[:, 0] = prior_centre
    m = {k: np.zeros_like(v) if not np.isscalar(v) else 0.0 for k, v in P.items()}
    for ep in range(epochs):
        G, _ = grads(P, X, Ap, An, y, l2, centre)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]
            P[k] = P[k] - lr * m[k]
    return P


def check_grad(seed=0):
    """Finite differences. Every network in this project ships a gradient check."""
    rng = np.random.default_rng(seed)
    n, F, d, k = 7, 6, 5, 3
    X = rng.normal(size=(n, F, d)); y = (rng.random(n) > 0.5).astype(float)
    Ap = np.abs(rng.normal(size=(F, F))) * 0.3; An = np.abs(rng.normal(size=(F, F))) * 0.3
    np.fill_diagonal(Ap, 0); np.fill_diagonal(An, 0)
    P = init(F, d, k, rng, None)
    centre = np.zeros((F, k))
    G, _ = grads(P, X, Ap, An, y, 0.05, centre)

    def loss(P):
        _, _, s = forward(P, X, Ap, An)
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
        ll = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()
        reg = 0.05 / 2 * sum(np.sum(P[q] ** 2) for q in ("W0", "Wp", "Wn", "b"))
        return ll + reg + 0.05 / 2 * np.sum((P["u"] - centre) ** 2)

    worst = 0.0
    for q in ("W0", "Wp", "Wn", "b", "u"):
        flat = P[q].ravel()
        for i in rng.choice(len(flat), min(6, len(flat)), replace=False):
            o = flat[i]; h = 1e-6
            flat[i] = o + h; lp = loss(P)
            flat[i] = o - h; lm = loss(P)
            flat[i] = o
            worst = max(worst, abs((lp - lm) / (2 * h) - G[q].ravel()[i]))
    return worst
