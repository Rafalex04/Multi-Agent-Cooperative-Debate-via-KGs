"""Multi-seed runner. Produces mean±std results and writes per-seed CSV."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# allow `python src/run.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_loaders
from src.model import build_model
from src.seed import set_seed
from src.train import train_one


DEFAULT_HPARAMS = {
    "hidden": 64,
    "num_bases": 8,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "patience": 30,
    "max_epochs": 200,
    "batch_size": 32,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="MUTAG", help="TUDataset name")
    p.add_argument("--model", default="rgcn", choices=["rgcn", "hinpool"], help="Model to run")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=None, help="Output CSV path (default: results/<dataset>_<model>.csv)")
    p.add_argument("--data_root", default="data/TU")
    p.add_argument("--splits_root", default="splits")
    p.add_argument("--hidden", type=int, default=DEFAULT_HPARAMS["hidden"])
    p.add_argument("--num_bases", type=int, default=DEFAULT_HPARAMS["num_bases"])
    p.add_argument("--dropout", type=float, default=DEFAULT_HPARAMS["dropout"])
    p.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_HPARAMS["weight_decay"])
    p.add_argument("--patience", type=int, default=DEFAULT_HPARAMS["patience"])
    p.add_argument("--max_epochs", type=int, default=DEFAULT_HPARAMS["max_epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_HPARAMS["batch_size"])
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--pool_ratio", type=float, default=0.9)
    p.add_argument("--attn_pool", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hparams = {
        "hidden": args.hidden,
        "num_bases": args.num_bases,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "max_epochs": args.max_epochs,
        "num_layers": args.num_layers,
        "pool_ratio": args.pool_ratio,
        "attn_pool": args.attn_pool,
    }

    out_path = args.out or f"results/{args.dataset}_{args.model}.csv"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # load data once (split is fixed, only model init varies)
    train_loader, val_loader, test_loader, meta = get_loaders(
        args.dataset,
        root=args.data_root,
        splits_root=args.splits_root,
        batch_size=args.batch_size,
    )
    hparams["num_classes"] = meta["num_classes"]
    print(f"Dataset: {args.dataset}  |  train/val/test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"Meta: {meta}")
    print(f"Hparams: {hparams}")
    print()

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        model = build_model(args.model, meta, hparams)
        print(f"[seed={seed}] training {args.model} on {args.dataset} ...", flush=True)
        test_auc, test_acc = train_one(
            model, train_loader, val_loader, test_loader, device, hparams
        )
        print(f"  → test AUROC: {test_auc:.4f}  acc: {test_acc:.4f}")
        rows.append({
            "dataset": args.dataset,
            "model": args.model,
            "seed": seed,
            "test_auc": test_auc,
            "test_acc": test_acc,
            **hparams,
        })

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    print()
    print(f"AUROC  {df.test_auc.mean():.4f} ± {df.test_auc.std():.4f}")
    print(f"Acc    {df.test_acc.mean():.4f} ± {df.test_acc.std():.4f}")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
