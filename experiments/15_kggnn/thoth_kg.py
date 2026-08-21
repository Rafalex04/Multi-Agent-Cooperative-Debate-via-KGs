"""ThothGNN re-derived: the knowledge graph IS the graph.

Keeps the parts of the Stage 4 architecture the ablation said were real, and
moves the parts it said were not.

  KEPT  anchored score        beta * logit(anchor) + gamma * graph, gamma init 0.
                              NO-ANCHOR at -0.0971 was the only measurable effect
                              in the whole ablation table, so the anchor stays and
                              is placed on the strongest available readout.
  KEPT  sign separation       separate weight matrices for edges joining findings
                              of the SAME KG polarity and of OPPOSITE polarity.
                              NO-SIGN was the only learned component that lost
                              accuracy when removed.
  KEPT  attention readout     bipolar attention over nodes.
  MOVED the KG                it was inert as extra nodes to aggregate from
                              (NO-KG-NODES +0.0049). Now it is the node set and
                              the edge set: nodes are the 16 findings, edges come
                              from lesions that present both.
  DROPPED claim nodes and     87% of those edges were DISAGREE and DISAGREE
          AGREE/DISAGREE      claims carry zero evidence. That information is
          edges               retained as the per-node `disputed` feature.
  DROPPED cross-layer fusion, one layer only; NO-XLAYER measured +0.0063 and
          learned edge gates  EDGE-FIXED +0.0042, i.e. both were better removed.

Forward pass:

    M = X W0 + (A+ X) W+ + (A- X) W- + b        A+ same polarity, A- opposite
    H = relu(M)
    a = softmax(H v)                            attention over findings
    z = sum_f a_f H_f
    s = beta * anchor + gamma * (z u + c)

Gradients are analytic and checked against finite differences by `check_grad()`,
which runs as part of the test entry point -- a hand-derived backward pass that
nobody verified is exactly how the Stage 4 bilinear dead point survived a full
ablation sweep.
"""
from __future__ import annotations

import numpy as np


def softmax(e, axis=-1):
    m = e - e.max(axis=axis, keepdims=True)
    x = np.exp(m)
    return x / x.sum(axis=axis, keepdims=True)


class ThothKG:
    def __init__(self, C, h, Ap, An, seed=0, gamma0=0.0, readout="attention"):
        """readout: 'attention' (softmax) or 'sum' (sigmoid gates, unnormalised).

        Softmax attention normalises the pooling weights to 1, which makes the
        readout a weighted MEAN and therefore blind to multiset cardinality --
        it cannot tell 3 malignant findings from 12. Sigmoid gates keep the same
        learned per-node emphasis but let the total grow with the evidence, which
        is the GIN argument for sum over mean.
        """
        self.readout = readout
        rng = np.random.default_rng(seed)
        sc = 1.0 / np.sqrt(C)
        self.W0 = rng.normal(0, sc, (C, h))
        self.Wp = rng.normal(0, sc, (C, h))
        self.Wn = rng.normal(0, sc, (C, h))
        self.b0 = np.zeros(h)
        self.v = rng.normal(0, 1.0 / np.sqrt(h), h)
        self.u = rng.normal(0, 1.0 / np.sqrt(h), h)
        self.c = 0.0
        self.beta = 1.0
        self.gamma = gamma0
        self.Ap, self.An = Ap, An

    # ---- parameter vector helpers, used by the gradient check ----
    _NAMES = ("W0", "Wp", "Wn", "b0", "v", "u", "c", "beta", "gamma")

    def get(self):
        return np.concatenate([np.atleast_1d(np.asarray(getattr(self, n))).ravel()
                               for n in self._NAMES])

    def set(self, vec):
        i = 0
        for n in self._NAMES:
            cur = np.asarray(getattr(self, n))
            k = max(1, cur.size)
            val = vec[i:i + k]
            setattr(self, n, float(val[0]) if cur.ndim == 0 else val.reshape(cur.shape))
            i += k

    def forward(self, X, anchor):
        Xp = np.einsum("ij,njc->nic", self.Ap, X)
        Xn = np.einsum("ij,njc->nic", self.An, X)
        M = X @ self.W0 + Xp @ self.Wp + Xn @ self.Wn + self.b0
        H = np.maximum(M, 0.0)
        e = H @ self.v
        a = (softmax(e, axis=1) if self.readout == "attention"
             else 1.0 / (1.0 + np.exp(-np.clip(e, -30, 30))))
        z = np.einsum("nf,nfh->nh", a, H)
        g = z @ self.u + self.c
        s = self.beta * anchor + self.gamma * g
        return s, dict(X=X, Xp=Xp, Xn=Xn, M=M, H=H, a=a, z=z, g=g, anchor=anchor)

    def backward(self, cache, dS, l2=0.0):
        X, Xp, Xn = cache["X"], cache["Xp"], cache["Xn"]
        M, H, a, z, g = cache["M"], cache["H"], cache["a"], cache["z"], cache["g"]
        gr = {}
        gr["gamma"] = float(dS @ g)
        gr["beta"] = float(dS @ cache["anchor"])
        dg = dS * self.gamma
        gr["u"] = z.T @ dg + l2 * self.u
        gr["c"] = float(dg.sum())
        dz = dg[:, None] * self.u                       # (n, h)
        dH = a[:, :, None] * dz[:, None, :]             # from z = sum a_f H_f
        da = np.einsum("nh,nfh->nf", dz, H)
        de = (a * (da - (a * da).sum(1, keepdims=True)) if self.readout == "attention"
              else da * a * (1.0 - a))                    # softmax / sigmoid backward
        dH = dH + de[:, :, None] * self.v[None, None, :]
        gr["v"] = np.einsum("nf,nfh->h", de, H)
        dM = dH * (M > 0)
        gr["W0"] = np.einsum("nfc,nfh->ch", X, dM) + l2 * self.W0
        gr["Wp"] = np.einsum("nfc,nfh->ch", Xp, dM) + l2 * self.Wp
        gr["Wn"] = np.einsum("nfc,nfh->ch", Xn, dM) + l2 * self.Wn
        gr["b0"] = dM.sum((0, 1))
        return gr

    def step(self, gr, lr):
        for n in self._NAMES:
            cur = getattr(self, n)
            if np.isscalar(cur) or np.asarray(cur).ndim == 0:
                setattr(self, n, float(cur - lr * gr[n]))
            else:
                setattr(self, n, cur - lr * gr[n])


def bce(s, y):
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    n = len(y)
    return float(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()), (p - y) / n


def fit(model, X, anchor, y, l2=1e-3, epochs=400, lr=0.05):
    for _ in range(epochs):
        s, cache = model.forward(X, anchor)
        _, dS = bce(s, y)
        model.step(model.backward(cache, dS, l2), lr)
    return model


def check_grad(seed=0, tol=2e-5, readout="attention"):
    """Finite-difference check of every analytic gradient."""
    rng = np.random.default_rng(seed)
    n, F, C, h = 6, 5, 4, 3
    X = rng.normal(size=(n, F, C))
    Ap = np.abs(rng.normal(size=(F, F))); Ap = (Ap + Ap.T) / 2; np.fill_diagonal(Ap, 0)
    An = np.abs(rng.normal(size=(F, F))); An = (An + An.T) / 2; np.fill_diagonal(An, 0)
    a0 = rng.normal(size=n)
    y = (rng.random(n) > 0.5).astype(float)
    m = ThothKG(C, h, Ap, An, seed=1, gamma0=0.7, readout=readout)

    s, cache = m.forward(X, a0)
    loss, dS = bce(s, y)
    gr = m.backward(cache, dS, 0.0)
    flat = np.concatenate([np.atleast_1d(np.asarray(gr[k])).ravel() for k in m._NAMES])

    p0 = m.get().copy()
    num = np.zeros_like(p0)
    eps = 1e-6
    for i in range(len(p0)):
        for sgn in (+1, -1):
            q = p0.copy(); q[i] += sgn * eps
            m.set(q)
            si, _ = m.forward(X, a0)
            li, _ = bce(si, y)
            num[i] += sgn * li
        num[i] /= (2 * eps)
    m.set(p0)
    err = np.abs(num - flat).max()
    print(f"gradient check [{readout:9s}]: max |analytic - numeric| = {err:.3e}  "
          f"{'PASS' if err < tol else 'FAIL'}")
    return err < tol


if __name__ == "__main__":
    ok = all(check_grad(readout=r) for r in ("attention", "sum"))
    raise SystemExit(0 if ok else 1)
