"""Multinomial head for ThothGNN v3 -- option B, for a NOMINAL label space.

RetinaMNIST is an ordinal ladder, so five grades gave four cut points and heads A
(cumulative-link) and C (stacked) both applied. DermaMNIST is SEVEN UNORDERED
classes: melanoma is not "more" than dermatofibroma, so a cut-point decomposition
is meaningless and the readout must be class-specific.

    H     = relu( X W0 + A+ X Wp + A- X Wn + b )          shared trunk, unchanged
    s[n,c] = sum_f sum_j u[f,j,c] H[n,f,j] + c0[c]        u is F x k x C now
    loss  = cross-entropy( softmax(s), y )

Everything above the readout is IDENTICAL to thothgnn3 -- same message passing,
same trunk shape, same optimiser -- so the comparison across datasets is of the
head, not of the architecture.

THE ONTOLOGY PRIOR IS RICHER HERE. On retina a finding's prior was a scalar
position on the ladder. Here it is a DISTRIBUTION over the seven classes, taken
from `maps_to_dermamnist_class`, and it becomes the L2 penalty centre of the
class-specific readout: centre[f, 0, c] = prior[f][c]. So the ontology states not
just "this finding matters" but "this finding indicates bcc", and the data has to
pay L2 to disagree.

Parameter count, F=55, d=4, k=2, C=7:  W0/Wp/Wn 3*4*2=24, b 2, u 55*2*7=770,
c0 7  ->  803. Larger than retina's 63 because the readout is per class.
"""
from __future__ import annotations

import numpy as np


def forward(P, X, Ap, An):
    """H = relu(X W0 + A+ X Wp + A- X Wn + b) ; s[n,c] = sum_fj u[f,j,c] H[n,f,j]"""
    M = X @ P["W0"] + np.einsum("fg,ngd->nfd", Ap, X) @ P["Wp"] \
        + np.einsum("fg,ngd->nfd", An, X) @ P["Wn"] + P["b"]
    H = np.maximum(M, 0.0)
    s = np.einsum("nfk,fkc->nc", H, P["u"]) + P["c0"]
    return H, M, s


def _softmax(s):
    z = s - s.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def grads(P, X, Ap, An, y, l2, u_centre):
    """Analytic gradient. Every network in this project ships a gradient check."""
    H, M, s = forward(P, X, Ap, An)
    n, C = s.shape
    p = _softmax(s)
    Y = np.zeros_like(p)
    Y[np.arange(n), np.asarray(y, int)] = 1.0
    e = (p - Y) / n                                   # n x C

    dH = np.einsum("nc,fkc->nfk", e, P["u"])
    dM = dH * (M > 0)
    ApX = np.einsum("fg,ngd->nfd", Ap, X)
    AnX = np.einsum("fg,ngd->nfd", An, X)
    G = {
        "W0": np.einsum("nfd,nfk->dk", X, dM) + l2 * P["W0"],
        "Wp": np.einsum("nfd,nfk->dk", ApX, dM) + l2 * P["Wp"],
        "Wn": np.einsum("nfd,nfk->dk", AnX, dM) + l2 * P["Wn"],
        "b": dM.sum(axis=(0, 1)) + l2 * P["b"],
        "u": np.einsum("nfk,nc->fkc", H, e) + l2 * (P["u"] - u_centre),
        "c0": e.sum(axis=0),                          # unregularised, like v3's c
    }
    return G, s


def init(F, d, k, C, rng):
    return {"W0": rng.normal(0, 0.3, (d, k)), "Wp": rng.normal(0, 0.3, (d, k)),
            "Wn": rng.normal(0, 0.3, (d, k)), "b": np.zeros(k),
            "u": rng.normal(0, 0.3, (F, k, C)), "c0": np.zeros(C)}


def fit(X, y, Ap, An, n_class, l2=0.1, kdim=4, epochs=800, lr=0.15, seed=0,
        prior_centre=None, class_weight=None):
    """Momentum SGD, full batch -- identical schedule to thothgnn3.fit.

    `class_weight` matters here in a way it never did on breast or retina:
    DermaMNIST train is 67% `nv` and 1.1% `df`, so an unweighted CE simply learns
    the majority. Pass 'balanced' to reweight by inverse class frequency.
    """
    F, d = X.shape[1], X.shape[2]
    rng = np.random.default_rng(seed)
    P = init(F, d, kdim, n_class, rng)
    centre = np.zeros((F, kdim, n_class))
    if prior_centre is not None:
        centre[:, 0, :] = np.asarray(prior_centre, float)
    y = np.asarray(y, int)
    if class_weight == "balanced":
        cnt = np.bincount(y, minlength=n_class).astype(float)
        w = np.where(cnt > 0, len(y) / (n_class * np.maximum(cnt, 1)), 0.0)
        keep = rng.choice(len(y), size=len(y), replace=True,
                          p=(w[y] / w[y].sum()))
        X, y = X[keep], y[keep]
    m = {q: np.zeros_like(v) for q, v in P.items()}
    for _ in range(epochs):
        G, _ = grads(P, X, Ap, An, y, l2, centre)
        for q in P:
            m[q] = 0.9 * m[q] + 0.1 * G[q]
            P[q] = P[q] - lr * m[q]
    return P


def predict(P, X, Ap, An):
    """-> (s logits, p class probabilities, argmax class)."""
    _, _, s = forward(P, X, Ap, An)
    p = _softmax(s)
    return s, p, p.argmax(axis=1)


def check_grad(seed=0, C=7):
    rng = np.random.default_rng(seed)
    n, F, d, k = 9, 6, 4, 3
    X = rng.normal(size=(n, F, d))
    y = rng.integers(0, C, size=n)
    Ap = np.abs(rng.normal(size=(F, F))) * 0.3
    An = np.abs(rng.normal(size=(F, F))) * 0.3
    np.fill_diagonal(Ap, 0); np.fill_diagonal(An, 0)
    P = init(F, d, k, C, rng)
    centre = np.zeros((F, k, C)); l2 = 0.05
    G, _ = grads(P, X, Ap, An, y, l2, centre)

    def loss(P):
        _, _, s = forward(P, X, Ap, An)
        p = _softmax(s)
        ll = -np.log(p[np.arange(n), y] + 1e-12).mean()
        reg = l2 / 2 * sum(np.sum(P[q] ** 2) for q in ("W0", "Wp", "Wn", "b"))
        return ll + reg + l2 / 2 * np.sum((P["u"] - centre) ** 2)

    worst = 0.0
    for q in ("W0", "Wp", "Wn", "b", "u", "c0"):
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
    print(f"gradient check (nominal, C=7): {w:.3e} {'PASS' if w < 1e-6 else 'FAIL'}")
