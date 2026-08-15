"""ResNet-18 supervised baseline on BreastMNIST.

Trains on the official MedMNIST split and evaluates on the held-out test set.
This is the reference point the VLM/debate results are compared against.

Protocol (MedMNIST v2 reference setup):
  - official splits only: 546 train / 78 val / 156 test, never re-split
  - ImageNet-pretrained ResNet-18, grayscale replicated to 3 channels
  - Adam lr 1e-3, batch 128, 100 epochs, lr x0.1 at epochs 50 and 75
  - checkpoint selected by best VALIDATION AUC, never final epoch
  - 5 seeds (0-4), mean +/- std reported

BreastMNIST is imbalanced (27% malignant), so AUC is primary and balanced
accuracy is reported alongside; raw accuracy is shown only for reference
against the 0.7308 majority baseline.

Label convention (MedMNIST v2): 0 -> MALIGNANT, 1 -> BENIGN.
MALIGNANT is the positive class, matching the zero-shot experiments.

Usage:
  python train_resnet.py --size 28 --seeds 0 1 2 3 4
  python train_resnet.py --size 28 --class-weighted
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
# MedMNIST label 0 is malignant; we model P(malignant) so the positive class
# matches the zero-shot experiments.
POS_LABEL_INT = 0


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training, and force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_split(split: str, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one official BreastMNIST split as (images NCHW float, labels long).

    Grayscale is replicated to 3 channels for the ImageNet stem and normalised
    with ImageNet statistics, matching the pretrained weights.
    """
    from medmnist import BreastMNIST

    ds = BreastMNIST(split=split, download=True, size=size)
    x = torch.from_numpy(ds.imgs).float().div_(255.0)          # (N,H,W)
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)                       # (N,3,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    # y = 1 when the sample is MALIGNANT
    y = torch.from_numpy((ds.labels[:, 0] == POS_LABEL_INT).astype("int64"))
    return x, y


def build_model() -> nn.Module:
    """ImageNet-pretrained ResNet-18 with a 2-class head."""
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, p_malignant) over a loader."""
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(probs)


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """AUC plus threshold metrics, with balanced accuracy for the imbalance."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")
    n = len(y_true)
    return {
        "auc": auc_score(y_true, y_prob),
        "accuracy": (tp + tn) / n if n else float("nan"),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "majority_baseline": max((y_true == 1).sum(), (y_true == 0).sum()) / n,
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }


def train_one_seed(seed: int, size: int, args) -> dict:
    """Train one model, select by best val AUC, return test metrics."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    xtr, ytr = load_split("train", size)
    xva, yva = load_split("val", size)
    xte, yte = load_split("test", size)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size,
                              shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(xva, yva), batch_size=256)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=256)

    model = build_model().to(device)

    if args.class_weighted:
        counts = torch.bincount(ytr, minlength=2).float()
        weight = (counts.sum() / (2 * counts)).to(device)
        logger.info("  class weights (benign, malignant): %s", weight.tolist())
    else:
        weight = None

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=args.milestones, gamma=0.1
    )

    best_val_auc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimiser.step()
        scheduler.step()

        yv, pv = evaluate(model, val_loader, device)
        val_auc = auc_score(yv, pv)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        if (epoch + 1) % 20 == 0:
            logger.info("    epoch %3d/%d  val_auc=%.4f  (best %.4f @ %d)",
                        epoch + 1, args.epochs, val_auc, best_val_auc, best_epoch)

    model.load_state_dict(best_state)
    yt, pt = evaluate(model, test_loader, device)
    m = metrics(yt, pt)
    m.update({
        "seed": seed,
        "size": size,
        "class_weighted": bool(args.class_weighted),
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "train_time_s": round(time.time() - t0, 1),
    })

    tag = f"{size}px{'_cw' if args.class_weighted else ''}"
    out_dir = RESULTS_DIR / tag / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["sample_id,split,y_true,y_prob,y_pred"]
    for i, (t, p) in enumerate(zip(yt, pt)):
        lines.append(f"{i:03d},test,{int(t)},{p:.6f},{int(p >= 0.5)}")
    (out_dir / "predictions.csv").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))

    logger.info("  seed %d: test AUC=%.4f acc=%.4f bal_acc=%.4f (best val AUC %.4f @ epoch %d, %.0fs)",
                seed, m["auc"], m["accuracy"], m["balanced_accuracy"],
                best_val_auc, best_epoch, m["train_time_s"])
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="ResNet-18 baseline on BreastMNIST")
    p.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--milestones", type=int, nargs="+", default=[50, 75])
    p.add_argument("--class-weighted", action="store_true")
    args = p.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    logger.info("ResNet-18 | %dpx | seeds %s | epochs %d | class_weighted=%s | device=%s",
                args.size, args.seeds, args.epochs, args.class_weighted,
                "cuda" if torch.cuda.is_available() else "cpu")

    runs = [train_one_seed(s, args.size, args) for s in args.seeds]

    tag = f"{args.size}px{'_cw' if args.class_weighted else ''}"
    summary = {"config": vars(args), "runs": runs, "aggregate": {}}
    for key in ("auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"):
        vals = [r[key] for r in runs if not np.isnan(r[key])]
        summary["aggregate"][key] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
        }

    out = RESULTS_DIR / tag / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    a = summary["aggregate"]
    print(f"\n{'='*62}\nResNet-18 {args.size}px | {len(runs)} seeds | class_weighted={args.class_weighted}\n{'='*62}")
    for key in ("auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1"):
        print(f"  {key:<20} {a[key]['mean']:.4f} +/- {a[key]['std']:.4f}"
              f"   [{a[key]['min']:.4f}, {a[key]['max']:.4f}]")
    print(f"\n  published ResNet-18 ({args.size}): AUC 0.901 / ACC 0.863"
          if args.size == 28 else "\n  published ResNet-18 (224): AUC 0.891 / ACC 0.833")
    print(f"  majority baseline: {runs[0]['majority_baseline']:.4f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
