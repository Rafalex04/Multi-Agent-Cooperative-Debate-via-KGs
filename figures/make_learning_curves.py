"""Render every learning-curve result file to a PNG.

Every number below was already computed and verified; this script only draws
it. Run from anywhere -- all paths are relative to the repo root, found by
walking up from this file.

  python figures/make_learning_curves.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                      "grid.alpha": 0.3})


def savefig(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p.relative_to(REPO)}")


# --------------------------------------------------------------- 1. breast ----
def breast_epoch_curves():
    """experiments/19_curves: train/test AUC vs epoch, 7 arms including ResNet."""
    f = REPO / "experiments/19_curves/results/train_curves.json"
    d = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, rows in d.items():
        ep = [r["epoch"] for r in rows]
        te = [r["test"] for r in rows]
        ax.plot(ep, te, marker="", label=label, linewidth=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("test AUC")
    ax.set_title("BreastMNIST -- test AUC vs training epoch")
    ax.legend(fontsize=7, loc="lower right")
    savefig(fig, "breast_epoch_curves.png")


def breast_debate_size_curve():
    """experiments/17_hetgnn: real-KG vs shuffled-KG AUC vs training set size."""
    f = REPO / "experiments/17_hetgnn/results/learning_curve.json"
    d = json.loads(f.read_text())
    rows = d["learning_curve"]
    n = [r["train_cases"] for r in rows]
    kg = [r["kg"] for r in rows]
    sh = [r["shuffled"] for r in rows]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4))
    a0.plot(n, kg, "o-", label="real KG", color="C0")
    a0.plot(n, sh, "o-", label="shuffled KG", color="C1")
    a0.set_xlabel("training cases"); a0.set_ylabel("test AUC")
    a0.set_title("BreastMNIST debate corpus\nreal vs shuffled KG"); a0.legend()
    gap = [r["gap"] for r in rows]
    a1.axhline(0, color="k", linewidth=0.8)
    a1.plot(n, gap, "o-", color="C2")
    a1.set_xlabel("training cases"); a1.set_ylabel("KG minus shuffled (AUC)")
    a1.set_title("gap vs training size")
    fig.tight_layout()
    savefig(fig, "breast_kg_vs_shuffled_curve.png")


# --------------------------------------------------------------- 2. retina ----
def retina_baseline_curve():
    """retinaMnist/results/baseline_curves.json: v3-A / v3-nograph / B2 / B1."""
    f = REPO / "retinaMnist/results/baseline_curves.json"
    d = json.loads(f.read_text())
    rows = d["curve"]
    n = [r["n_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    style = {"v3": ("ThothGNN v3", "C0"), "v3ng": ("v3, no graph (A=0)", "C1"),
             "b2": ("B2 GraphGeo", "C2"), "b1": ("B1 Catfish", "C3")}
    for key, (label, color) in style.items():
        mean = np.array([r[key][0] for r in rows])
        sd = np.array([r[key][1] for r in rows])
        ax.plot(n, mean, "o-", color=color, label=label)
        ax.fill_between(n, mean - sd, mean + sd, color=color, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("training images (log scale)"); ax.set_ylabel("referable-DR AUC")
    ax.set_title("RetinaMNIST -- learning curve, 3 seeds, shaded ± 1 sd")
    ax.legend(fontsize=8)
    savefig(fig, "retina_baseline_curve.png")


# --------------------------------------------------------------- 3. derma -----
def derma_curve(name, title, out_name):
    f = REPO / f"dermaMnist/results/{name}"
    d = json.loads(f.read_text())
    rows = d["curve"]
    n = [r["n_train"] for r in rows]
    real = np.array([r["macro_recall"][0] for r in rows])
    real_sd = np.array([r["macro_recall"][1] for r in rows])
    sh = np.array([r["shuffled"][0] for r in rows])
    sh_sd = np.array([r["shuffled"][1] for r in rows])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4))
    a0.plot(n, real, "o-", color="C0", label="real KG")
    a0.fill_between(n, real - real_sd, real + real_sd, color="C0", alpha=0.15)
    a0.plot(n, sh, "o-", color="C1", label="shuffled KG")
    a0.fill_between(n, sh - sh_sd, sh + sh_sd, color="C1", alpha=0.15)
    a0.set_xscale("log")
    a0.set_xlabel("training images (log scale)"); a0.set_ylabel("macro-recall")
    a0.set_title(title); a0.legend(fontsize=8)

    delta = np.array([r["delta"][0] for r in rows])
    delta_sd = np.array([r["delta"][1] for r in rows])
    a1.axhline(0, color="k", linewidth=0.8)
    a1.errorbar(n, delta, yerr=delta_sd, fmt="o-", color="C2", capsize=3)
    a1.set_xscale("log")
    a1.set_xlabel("training images (log scale)")
    a1.set_ylabel("real − shuffled (macro-recall)")
    slope = d["delta_slope_vs_logn"]
    a1.set_title(f"KG − shuffled vs log n\nslope {slope:+.2e}")
    fig.tight_layout()
    savefig(fig, out_name)


# ----------------------------------------------------------- 4. gap-law -------
def transfer_gap_law():
    """experiments/22_transfer/results/gap_law.json: in-domain AUC predicts the
    external transfer gap across six perception-layer variants (C4)."""
    f = REPO / "experiments/22_transfer/results/gap_law.json"
    d = json.loads(f.read_text())
    blocks = d["blocks"]
    x = np.array([b["indomain"] for b in blocks])
    y = np.array([b["gap"] for b in blocks])
    labels = [b["label"] for b in blocks]
    slope, intercept = d["slope"], d["intercept"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, color="C0", zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=7, xytext=(5, 5),
                    textcoords="offset points")
    xs = np.linspace(x.min() - 0.01, x.max() + 0.01, 50)
    ax.plot(xs, slope * xs + intercept, "--", color="C3",
            label=f"slope {slope:.3f} (95% CI computed separately)")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("in-domain AUC (BreastMNIST test, n=156)")
    ax.set_ylabel("external gap (in-domain − BUS-BRA, n=1064 patients)")
    ax.set_title(f"Transfer-gap law across perception variants\n"
                 f"r={d['pearson_indomain_gap']:.4f}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "transfer_gap_law.png")


if __name__ == "__main__":
    print("Rendering learning-curve figures to figures/ ...")
    breast_epoch_curves()
    breast_debate_size_curve()
    retina_baseline_curve()
    derma_curve("derma_curve.json", "DermaMNIST (probes + debate)\nreal vs shuffled KG",
                "derma_curve_with_debate.png")
    derma_curve("derma_curve_probeonly.json", "DermaMNIST (probes only, no debate)\nreal vs shuffled KG",
                "derma_curve_probeonly.png")
    transfer_gap_law()
    print("Done.")
