import copy

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.loader import DataLoader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    loss_fn,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train() if train else model.eval()
    all_labels, all_probs = [], []

    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            logits = model(batch)
            loss = loss_fn(logits, batch.y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if logits.size(-1) == 2:
            probs = logits.softmax(dim=-1)[:, 1]
        else:
            probs = logits.softmax(dim=-1).max(dim=-1).values

        all_labels.extend(batch.y.cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    if len(set(all_labels)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(all_labels, all_probs)

    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    return auc, acc


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    hparams: dict,
) -> tuple[float, float]:
    """Train with early stopping on val AUROC. Returns (test_auc, test_acc) at best-val epoch."""
    model = model.to(device)

    # compute pos_weight for binary imbalance
    labels = [int(b.y.item()) for batch in train_loader for b in batch.to_data_list()]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device) if n_pos > 0 else None

    if hparams.get("num_classes", 2) == 2 and pos_weight is not None:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def loss_wrapper(logits, y):
            return loss_fn(logits[:, 1].float(), y.float())
    else:
        loss_fn = nn.CrossEntropyLoss()
        loss_wrapper = loss_fn

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.get("lr", 1e-3),
        weight_decay=hparams.get("weight_decay", 5e-4),
    )

    patience = hparams.get("patience", 30)
    max_epochs = hparams.get("max_epochs", 200)

    best_val_auc = -1.0
    best_test_auc = float("nan")
    best_test_acc = float("nan")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        _run_epoch(model, train_loader, optimizer, loss_wrapper, device, train=True)
        val_auc, _ = _run_epoch(model, val_loader, optimizer, loss_wrapper, device, train=False)
        test_auc, test_acc = _run_epoch(model, test_loader, optimizer, loss_wrapper, device, train=False)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_test_acc = test_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_test_auc, best_test_acc
