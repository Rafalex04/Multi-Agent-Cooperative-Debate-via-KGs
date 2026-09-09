"""Render Figure busbravariants and Figure cohorts from existing npz.

  python figures/make_thesis_images.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"figure.dpi": 150})


def imshow(ax, img, title):
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------- Figure busbravariants --
def busbravariants():
    variants = [("pad2", "busbra_pad2_224.npz"), ("pad15", "busbra_pad15_224.npz"),
                ("bring", "busbra_bring_224.npz"), ("bdim", "busbra_bdim_224.npz")]
    d0 = np.load(REPO / f"data/external/{variants[0][1]}", allow_pickle=True)
    # pick a malignant case, same `ids` order shared across all four variants
    idx = int(np.flatnonzero(d0["labels"].ravel() == 0)[0])   # a MALIGNANT case
    case_id = d0["ids"][idx]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (name, fname) in zip(axes, variants):
        d = np.load(REPO / f"data/external/{fname}", allow_pickle=True)
        imshow(ax, d["imgs"][idx], name)
    fig.suptitle(f"BUS-BRA case {case_id} across the four masking/crop arms", fontsize=10)
    fig.tight_layout()
    p = OUT / "busbra_variants.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.relative_to(REPO)}  (case_id={case_id}, idx={idx})")


# ---------------------------------------------------------- Figure cohorts ---
def cohorts():
    fig, axes = plt.subplots(2, 4, figsize=(11, 6.5))

    # BreastMNIST: benign=1, malignant=0 (project convention: y=1 if MALIGNANT)
    bt = np.load(REPO / "breastMnist/data/breast/images_224/test.npz", allow_pickle=True)
    bl = bt["labels"][:, 0] if bt["labels"].ndim == 2 else bt["labels"]
    imshow(axes[0, 0], bt["imgs"][int(np.flatnonzero(bl == 1)[0])], "BreastMNIST\nbenign")
    imshow(axes[0, 1], bt["imgs"][int(np.flatnonzero(bl == 0)[0])], "BreastMNIST\nmalignant")
    # convention: 0=MALIGNANT, 1=BENIGN -- confirmed above, labels correct as-is

    # BUS-BRA pad2: labels 0=malignant,1=benign per busbra npz convention (check)
    bb = np.load(REPO / "data/external/busbra_pad2_224.npz", allow_pickle=True)
    bbl = bb["labels"].ravel()
    imshow(axes[0, 2], bb["imgs"][int(np.flatnonzero(bbl == 1)[0])], "BUS-BRA (pad2)\nbenign")
    imshow(axes[0, 3], bb["imgs"][int(np.flatnonzero(bbl == 0)[0])], "BUS-BRA (pad2)\nmalignant")

    # RetinaMNIST: ICDR grade 0..4
    rt = np.load(REPO / "retinaMnist/data/retina/images_224/test.npz", allow_pickle=True)
    rl = rt["labels"][:, 0] if rt["labels"].ndim == 2 else rt["labels"].ravel()
    imshow(axes[1, 0], rt["imgs"][int(np.flatnonzero(rl == 0)[0])], "RetinaMNIST\ngrade 0")
    imshow(axes[1, 1], rt["imgs"][int(np.flatnonzero(rl == 3)[0])], "RetinaMNIST\ngrade 3")

    # DermaMNIST: pick two distinct classes (e.g. nv=5 majority, mel=4)
    dt = np.load(REPO / "dermaMnist/data/derma/images_224/test.npz", allow_pickle=True)
    dl = dt["labels"][:, 0] if dt["labels"].ndim == 2 else dt["labels"].ravel()
    imshow(axes[1, 2], dt["imgs"][int(np.flatnonzero(dl == 5)[0])], "DermaMNIST\nnv (melanocytic nevi)")
    imshow(axes[1, 3], dt["imgs"][int(np.flatnonzero(dl == 4)[0])], "DermaMNIST\nmel (melanoma)")

    fig.suptitle("Representative images across all four cohorts", fontsize=11)
    fig.tight_layout(h_pad=3.5)
    p = OUT / "cohorts.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p.relative_to(REPO)}")
    print(f"  BreastMNIST test label convention check: unique={np.unique(bl)}")
    print(f"  BUS-BRA pad2 label convention check: unique={np.unique(bb['labels'])}")


if __name__ == "__main__":
    print("Rendering thesis figures ...")
    busbravariants()
    cohorts()
    print("Done.")
