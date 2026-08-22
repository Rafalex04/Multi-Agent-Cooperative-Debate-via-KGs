"""S1 / S6 -- the lesion-layer network.

Hypothesis. The clinically correct inference is findings -> lesion -> malignancy.
The ontology already contains that middle layer, and its "lesion presents finding"
relation is a ready-made sparsity mask. A flat per-finding logistic cannot compute
"this finding pattern matches fibroadenoma", because that is a CONJUNCTION over a
specific subset of findings, not a weighted sum over all of them.

Why this escapes what has already been falsified. Nothing propagates between
findings, so per-finding discrimination is never mixed -- that was the failure of
every previous KG-topology attempt. The pass is directed and bipartite, findings
send and lesions receive, exactly once.

G0 supports the prior: restricting pairwise probe products to ontology-adjacent
pairs beat both no products (CV +0.0067) and ALL products (CV +0.0098). The mask
is doing work. S1 is the higher-order version of the same idea -- a lesion
presents up to seven findings, so it expresses conjunctions a pairwise product
cannot.

    x_f = X_f . u                       shared projection, C params
    z_l = sum_f M[l,f] w[l,f] x_f + b_l  masked, 43 nonzero weights
    h   = relu(z)                        bias is a threshold, so h is a soft AND
    g   = sum_l sign_l h_l               sign fixed by the KG's lesion class
    s   = beta logit(anchor) + gamma g   gamma init 0

Order 10^2 parameters, every one interpretable as "how much finding f evidences
lesion l".

CAVEAT recorded up front: this ontology's lesion catalogue is one-sided. Fifteen
of sixteen classifiable lesions are benign, because the graph describes benign
entities by appearance and describes malignancy through stance triples instead.
So the lesion layer is a BENIGN-PATTERN detector, not a differential. That is a
fact about the knowledge graph, not a modelling choice, and the write-up must
say so.

S6 is the same computation written independently as a masked MLP. The two
forward passes are asserted equal to 1e-6, which is an independent-implementation
correctness gate on top of the finite-difference gradient check.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "14_kgtensor"))
from claims import SPLITS, auc, bacc, kg_findings                     # noqa: E402
from features import COLS, build                                      # noqa: E402
from kg_structure import build as build_structure                     # noqa: E402

_KG = _HERE.parents[2] / "breastMnist/data/breast/knowledge_graph.json"
_D = _HERE.parents[2] / "breastMnist/data/breast"


class LesionNet:
    _N = ("u", "W", "b", "beta", "gamma")

    def __init__(self, C, mask, sign, seed=0, gamma0=0.0):
        rng = np.random.default_rng(seed)
        self.M = (mask > 0).astype(float)
        self.sign = sign
        self.u = rng.normal(0, 1.0 / np.sqrt(C), C)
        self.W = mask.copy()                    # init from Adamic-Adar weights
        self.b = np.zeros(len(sign))
        self.beta, self.gamma = 1.0, gamma0

    def get(self):
        return np.concatenate([np.atleast_1d(np.asarray(getattr(self, n))).ravel()
                               for n in self._N])

    def set(self, v):
        i = 0
        for n in self._N:
            cur = np.asarray(getattr(self, n)); k = max(1, cur.size)
            val = v[i:i + k]
            setattr(self, n, float(val[0]) if cur.ndim == 0 else val.reshape(cur.shape))
            i += k

    def forward(self, X, anchor):
        x = np.einsum("nfc,c->nf", X, self.u)
        Wm = self.W * self.M
        z = x @ Wm.T + self.b
        h = np.maximum(z, 0.0)
        g = h @ self.sign
        s = self.beta * anchor + self.gamma * g
        return s, dict(X=X, x=x, z=z, h=h, g=g, Wm=Wm, anchor=anchor)

    def backward(self, c, dS, l2=0.0):
        gr = {}
        gr["gamma"] = float(dS @ c["g"])
        gr["beta"] = float(dS @ c["anchor"])
        dg = dS * self.gamma
        dh = dg[:, None] * self.sign[None, :]
        dz = dh * (c["z"] > 0)
        gr["W"] = (dz.T @ c["x"]) * self.M + l2 * self.W * self.M
        gr["b"] = dz.sum(0)
        dx = dz @ c["Wm"]
        gr["u"] = np.einsum("nf,nfc->c", dx, c["X"]) + l2 * self.u
        return gr

    def step(self, gr, lr):
        for n in self._N:
            cur = getattr(self, n)
            if np.isscalar(cur) or np.asarray(cur).ndim == 0:
                setattr(self, n, float(cur - lr * gr[n]))
            else:
                setattr(self, n, cur - lr * gr[n])


def forward_masked_mlp(m, X, anchor):
    """S6: the same computation, written independently, loop-based."""
    n, F, C = X.shape
    L = len(m.sign)
    out = np.zeros(n)
    for i in range(n):
        xf = np.array([float(X[i, f] @ m.u) for f in range(F)])
        g = 0.0
        for l in range(L):
            acc = m.b[l]
            for f in range(F):
                if m.M[l, f]:
                    acc += m.W[l, f] * xf[f]
            g += m.sign[l] * max(acc, 0.0)
        out[i] = m.beta * anchor[i] + m.gamma * g
    return out


def bce(s, y):
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    return (p - y) / len(y)


def rank_grad(s, y, npairs=512, rng=None):
    rng = rng or np.random.default_rng(0)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    i, j = rng.choice(pos, npairs), rng.choice(neg, npairs)
    g = -1.0 / (1.0 + np.exp(np.clip(s[i] - s[j], -30, 30)))
    e = np.zeros(len(y)); np.add.at(e, i, g); np.add.at(e, j, -g)
    return e / npairs


def fit(m, X, a, y, l2=1e-2, epochs=800, lr=0.05, loss="ce", seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        s, c = m.forward(X, a)
        d = bce(s, y) if loss == "ce" else rank_grad(s, y, rng=rng)
        m.step(m.backward(c, d, l2), lr)
    return m


def check_grad(tol=2e-5):
    rng = np.random.default_rng(0)
    n, F, C, L = 7, 6, 3, 4
    X = rng.normal(size=(n, F, C))
    mask = (rng.random((L, F)) > 0.5) * rng.random((L, F))
    sign = rng.choice([-1.0, 1.0], L)
    a = rng.normal(size=n); y = (rng.random(n) > .5).astype(float)
    m = LesionNet(C, mask, sign, seed=1, gamma0=0.6)
    s, c = m.forward(X, a)
    gr = m.backward(c, bce(s, y), 0.0)
    flat = np.concatenate([np.atleast_1d(np.asarray(gr[k])).ravel() for k in m._N])
    p0 = m.get().copy(); num = np.zeros_like(p0); eps = 1e-6
    for i in range(len(p0)):
        acc = 0.0
        for sg in (+1, -1):
            q = p0.copy(); q[i] += sg * eps; m.set(q)
            si, _ = m.forward(X, a)
            p = 1.0 / (1.0 + np.exp(-np.clip(si, -30, 30)))
            acc += sg * float(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean())
        num[i] = acc / (2 * eps)
    m.set(p0)
    err = np.abs(num - flat).max()
    print(f"  gradient check      max |analytic - numeric| = {err:.2e}  "
          f"{'PASS' if err < tol else 'FAIL'}")
    # S6 independent-implementation gate
    s1, _ = m.forward(X, a)
    s6 = forward_masked_mlp(m, X, a)
    e2 = np.abs(s1 - s6).max()
    print(f"  S1 vs S6 forward    max |difference|          = {e2:.2e}  "
          f"{'PASS' if e2 < 1e-6 else 'FAIL'}")
    return err < tol and e2 < 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["probe", "probeneg"])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=== correctness gates ===")
    if not check_grad():
        raise SystemExit("correctness gate failed")

    runs = [str(_D / r) for r in ("debates_v5q", "debates_v5q_r1",
                                  "debates_v5q_r2", "debates_v5q_r3")]
    findings = kg_findings()
    blocks = []
    for tag in args.tags:
        rows, Xt, Y, _ = build(runs, tag=tag)
        blocks.append(Xt)
    st = build_structure(_KG, findings)
    mask, sign = st["mask"], st["lesion_sign"]
    print(f"\nlesion layer: {mask.shape[0]} lesions, {int((mask>0).sum())} edges, "
          f"MAL {int((sign>0).sum())} / BEN {int((sign<0).sum())}")

    dc = [COLS.index(c) for c in ("net", "presence", "mass", "endorsed", "disputed", "open")]
    X = {}
    for s in SPLITS:
        probes = [b[s][:, :, 0:1] for b in blocks]
        X[s] = np.concatenate(probes + [blocks[0][s][:, :, dc]], axis=2)
    C = X["train"].shape[2]
    mu, sd = X["train"].mean((0, 1)), X["train"].std((0, 1))
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xz = {s: (X[s] - mu) / sd for s in SPLITS}
    print(f"node features {C}  ({len(blocks)} probe phrasings + {len(dc)} debate)")

    # ---- anchor: flat probe model over all phrasings, train-only ----
    from gates import fit_ce, cv_score, full_eval, L2S
    A = {s: np.hstack([b[s][:, :, 0] for b in blocks]) for s in SPLITS}
    cv_a, l2a = max(((cv_score(A["train"], Y["train"], l2, fit_ce)[0], l2) for l2 in L2S))
    aa, ab, asc = full_eval(A, Y, l2a, fit_ce)
    print(f"\nANCHOR flat probe-{A['train'].shape[1]}: TEST {aa['test']:.4f} "
          f"bAcc {ab:.4f}  (cv {cv_a:.4f})")
    anchor = asc

    res = {"anchor": {"test": aa["test"], "bacc": ab, "cv": cv_a}}
    print(f"\n  {'loss':6s} {'l2':>6s}  {'CV':>7s}  {'TEST':>16s} {'bAcc':>16s} {'gamma':>7s}")
    ytr = Y["train"]
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(ytr)), 5)
    best = None
    for loss in ("ce", "rank"):
        for l2 in (1e-3, 1e-2, 1e-1):
            sc = np.zeros(len(ytr))
            for fo in folds:
                mk = np.zeros(len(ytr), dtype=bool); mk[fo] = True
                mm = LesionNet(C, mask, sign, seed=0, gamma0=0.0)
                fit(mm, Xz["train"][~mk], anchor["train"][~mk], ytr[~mk],
                    l2, args.epochs, loss=loss)
                sc[mk] = mm.forward(Xz["train"][mk], anchor["train"][mk])[0]
            cv = auc(list(sc[ytr == 1]), list(sc[ytr == 0]))
            ts, bs, gs = [], [], []
            for seed in range(args.seeds):
                mm = LesionNet(C, mask, sign, seed=seed, gamma0=0.0)
                fit(mm, Xz["train"], anchor["train"], ytr, l2, args.epochs, loss=loss)
                s_ = {k: mm.forward(Xz[k], anchor[k])[0] for k in SPLITS}
                ts.append(auc(list(s_["test"][Y["test"] == 1]), list(s_["test"][Y["test"] == 0])))
                sf = np.concatenate([s_["train"], s_["val"]])
                yf = np.concatenate([Y["train"], Y["val"]])
                t = max(sorted(set(sf.tolist())), key=lambda t: bacc(sf.tolist(), yf.tolist(), t))
                bs.append(bacc(s_["test"].tolist(), Y["test"].tolist(), t)); gs.append(mm.gamma)
            print(f"  {loss:6s} {l2:6g}  {cv:7.4f}  {np.mean(ts):.4f}+-{np.std(ts):.4f} "
                  f"{np.mean(bs):.4f}+-{np.std(bs):.4f} {np.mean(gs):+7.3f}")
            res[f"S1|{loss}|l2={l2}"] = {"cv": cv, "test": float(np.mean(ts)),
                                         "test_sd": float(np.std(ts)),
                                         "bacc": float(np.mean(bs)),
                                         "gamma": float(np.mean(gs))}
            if best is None or cv > best[0]:
                best = (cv, loss, l2, float(np.mean(ts)), float(np.mean(bs)))
    print(f"\n  CV-selected: loss={best[1]} l2={best[2]}  ->  TEST {best[3]:.4f}  bAcc {best[4]:.4f}")
    print(f"  KILL CRITERION: CV {best[0]:.4f} vs anchor CV {cv_a:.4f}  -> "
          + ("SURVIVES" if best[0] > cv_a else "KILLED"))
    res["selected"] = {"loss": best[1], "l2": best[2], "cv": best[0],
                       "test": best[3], "bacc": best[4],
                       "survives": bool(best[0] > cv_a)}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
