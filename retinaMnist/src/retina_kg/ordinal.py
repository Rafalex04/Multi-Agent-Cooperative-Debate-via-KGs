"""Ordinal heads for ThothGNN v3: option A (cumulative-link) and option C (stacked).

RetinaMNIST is a 5-level ICDR severity scale, and v3 ends in a single scalar
followed by a sigmoid. Five ordered grades have FOUR cut points, not five, so
both options below decompose the problem the same way -- into the four questions
"is the grade above c?" for c = 0,1,2,3. They differ only in parameter sharing:

  A  cumulative-link (CORAL).  ONE trunk, four cut points.
       P(y > c) = sigmoid(s - theta_c),  s = sum_f u_f . H_f   (no bias; theta carries it)
       trunk 59 params + 4 = 63 for F=16, d=4, k=2.

  C  stacked binary heads.     FOUR independent trunks.
       head c is an unmodified thothgnn3 fitted on y_c = 1[y > c].
       4 x 59 = 236 params, nothing shared.

So A is C with proportional odds imposed. If they score the same, one shared
representation suffices; if C wins, the findings separating grade 0 from 1 are
genuinely different from those separating 3 from 4 -- which is what the ICDR
ladder claims (microaneurysms at the bottom, neovascularisation at the top).

EXACT REDUCTION. With C=2 there is one cut point, and A becomes thothgnn3 with
`c` reparameterised as `-theta_0`. Both are unregularised, so with theta
initialised at zero the two trajectories are mirror images and the fitted scores
agree to floating point. `test_reduction()` asserts this against the real breast
corpus -- it is the regression test that keeps this file honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import add_experiment_paths                              # noqa: E402
add_experiment_paths("17_hetgnn")
import thothgnn3                                                    # noqa: E402
from thothgnn3 import forward as _v3_forward                        # noqa: E402


# ------------------------------------------------------------- option A: CORAL --
def forward(P, X, Ap, An):
    """H = relu(X W0 + A+ X Wp + A- X Wn + b) ; s = sum_f H_f . u   (NO bias).

    The bias lives in theta so that the C=2 case is exactly thothgnn3.
    """
    M = X @ P["W0"] + np.einsum("fg,ngd->nfd", Ap, X) @ P["Wp"] \
        + np.einsum("fg,ngd->nfd", An, X) @ P["Wn"] + P["b"]
    H = np.maximum(M, 0.0)
    return H, M, (H * P["u"]).sum(axis=(1, 2))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def cum_targets(y, n_cut):
    """t[n, c] = 1 if grade_n > c else 0."""
    return (np.asarray(y)[:, None] > np.arange(n_cut)[None, :]).astype(float)


def grads(P, X, Ap, An, y, l2, u_centre, n_cut):
    """Analytic gradient. Every network in this project ships a gradient check."""
    H, M, s = forward(P, X, Ap, An)
    T = cum_targets(y, n_cut)
    p = _sigmoid(s[:, None] - P["theta"][None, :])
    e = (p - T) / len(y)                       # n x n_cut
    E = e.sum(axis=1)                          # dL/ds_n, summed over cut points

    dH = E[:, None, None] * P["u"]
    dM = dH * (M > 0)
    ApX = np.einsum("fg,ngd->nfd", Ap, X)
    AnX = np.einsum("fg,ngd->nfd", An, X)
    G = {
        "W0": np.einsum("nfd,nfk->dk", X, dM) + l2 * P["W0"],
        "Wp": np.einsum("nfd,nfk->dk", ApX, dM) + l2 * P["Wp"],
        "Wn": np.einsum("nfd,nfk->dk", AnX, dM) + l2 * P["Wn"],
        "b": dM.sum(axis=(0, 1)) + l2 * P["b"],
        "u": np.einsum("n,nfk->fk", E, H) + l2 * (P["u"] - u_centre),
        "theta": -e.sum(axis=0),               # unregularised, like v3's `c`
    }
    return G, s


def init(F, d, k, n_cut, rng):
    """theta starts at zero: that is what makes the C=2 reduction exact."""
    return {"W0": rng.normal(0, 0.3, (d, k)), "Wp": rng.normal(0, 0.3, (d, k)),
            "Wn": rng.normal(0, 0.3, (d, k)), "b": np.zeros(k),
            "u": rng.normal(0, 0.3, (F, k)), "theta": np.zeros(n_cut)}


def fit(X, y, Ap, An, n_cut, l2=0.1, kdim=4, epochs=800, lr=0.15, seed=0,
        prior_centre=None):
    """Momentum SGD, full batch -- identical schedule to thothgnn3.fit."""
    F, d = X.shape[1], X.shape[2]
    rng = np.random.default_rng(seed)
    P = init(F, d, kdim, n_cut, rng)
    centre = np.zeros((F, kdim))
    if prior_centre is not None:
        centre[:, 0] = prior_centre
    m = {q: np.zeros_like(v) for q, v in P.items()}
    for _ in range(epochs):
        G, _ = grads(P, X, Ap, An, y, l2, centre, n_cut)
        for q in P:
            m[q] = 0.9 * m[q] + 0.1 * G[q]
            P[q] = P[q] - lr * m[q]
    return P


def predict(P, X, Ap, An):
    """-> (s, p[n, n_cut], grade_hat). grade = how many cut points s clears."""
    _, _, s = forward(P, X, Ap, An)
    p = _sigmoid(s[:, None] - P["theta"][None, :])
    return s, p, (p > 0.5).sum(axis=1)


# --------------------------------------------------- option C: stacked binary --
def fit_stacked(X, y, Ap, An, n_cut, l2=0.1, kdim=4, epochs=800, lr=0.15, seed=0,
                prior_centre=None):
    """Four unmodified thothgnn3 models, one per cut point. Nothing is shared."""
    T = cum_targets(y, n_cut)
    return [thothgnn3.fit(X, T[:, c], Ap, An, l2=l2, kdim=kdim, epochs=epochs,
                          lr=lr, seed=seed, prior_centre=prior_centre)
            for c in range(n_cut)]


def predict_stacked(Ps, X, Ap, An):
    """-> (S[n, n_cut], p[n, n_cut], grade_hat). Each head has its own scalar."""
    S = np.stack([_v3_forward(P, X, Ap, An)[2] for P in Ps], axis=1)
    p = _sigmoid(S)
    return S, p, (p > 0.5).sum(axis=1)


def rank_violations(p):
    """Samples whose P(y>c) is not non-increasing in c.

    A is rank-consistent by construction when theta comes out sorted; C has four
    unrelated trunks and nothing enforces it, so this is reported, not assumed.
    """
    return int((np.diff(p, axis=1) > 1e-9).any(axis=1).sum())


# ------------------------------------------------------------- gradient check --
def check_grad(seed=0, n_cut=4):
    rng = np.random.default_rng(seed)
    n, F, d, k = 9, 6, 5, 3
    X = rng.normal(size=(n, F, d))
    y = rng.integers(0, n_cut + 1, size=n).astype(float)
    Ap = np.abs(rng.normal(size=(F, F))) * 0.3
    An = np.abs(rng.normal(size=(F, F))) * 0.3
    np.fill_diagonal(Ap, 0); np.fill_diagonal(An, 0)
    P = init(F, d, k, n_cut, rng)
    P["theta"] = rng.normal(0, 0.5, n_cut)          # not at the symmetric point
    centre = np.zeros((F, k))
    l2 = 0.05
    G, _ = grads(P, X, Ap, An, y, l2, centre, n_cut)

    def loss(P):
        _, _, s = forward(P, X, Ap, An)
        T = cum_targets(y, n_cut)
        p = _sigmoid(s[:, None] - P["theta"][None, :])
        ll = -(T * np.log(p + 1e-12) + (1 - T) * np.log(1 - p + 1e-12)).sum(axis=1).mean()
        reg = l2 / 2 * sum(np.sum(P[q] ** 2) for q in ("W0", "Wp", "Wn", "b"))
        return ll + reg + l2 / 2 * np.sum((P["u"] - centre) ** 2)

    worst = 0.0
    for q in ("W0", "Wp", "Wn", "b", "u", "theta"):
        flat = P[q].ravel()
        for i in rng.choice(len(flat), min(6, len(flat)), replace=False):
            o = flat[i]; h = 1e-6
            flat[i] = o + h; lp = loss(P)
            flat[i] = o - h; lm = loss(P)
            flat[i] = o
            worst = max(worst, abs((lp - lm) / (2 * h) - G[q].ravel()[i]))
    return worst


if __name__ == "__main__":
    w = check_grad()
    print(f"gradient check (ordinal, 4 cut points): {w:.3e} "
          f"{'PASS' if w < 1e-6 else 'FAIL'}")
