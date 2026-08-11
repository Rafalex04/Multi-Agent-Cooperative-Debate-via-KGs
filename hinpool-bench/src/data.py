import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


def load_dataset(name: str, root: str = "data/TU") -> list:
    ds = TUDataset(root=root, name=name)
    processed = []
    for data in ds:
        data = data.clone()
        if data.x is not None and data.x.size(1) > 1:
            data.node_type = data.x.argmax(dim=1)
        else:
            data.node_type = torch.zeros(data.num_nodes, dtype=torch.long)

        if getattr(data, "edge_attr", None) is not None and data.edge_attr.dim() > 1:
            data.edge_type = data.edge_attr.argmax(dim=1)
        else:
            data.edge_type = torch.zeros(data.num_edges, dtype=torch.long)

        processed.append(data)
    return processed


def get_split(labels: list, name: str, seed: int = 0, root: str = "splits") -> dict:
    """Create fixed 80/10/10 stratified split, persisted to disk and reused."""
    Path(root).mkdir(parents=True, exist_ok=True)
    path = Path(root) / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    idx = np.arange(len(labels))
    labels_arr = np.array(labels)
    train, tmp = train_test_split(idx, test_size=0.2, stratify=labels_arr, random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, stratify=labels_arr[tmp], random_state=seed)
    split = {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}
    with open(path, "w") as f:
        json.dump(split, f)
    return split


def get_loaders(
    name: str,
    root: str = "data/TU",
    splits_root: str = "splits",
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    dataset = load_dataset(name, root=root)
    labels = [int(d.y.item()) for d in dataset]
    split = get_split(labels, name, root=splits_root)

    train_data = [dataset[i] for i in split["train"]]
    val_data = [dataset[i] for i in split["val"]]
    test_data = [dataset[i] for i in split["test"]]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # dataset meta — needed to build the model
    sample = dataset[0]
    meta = {
        "num_node_features": sample.x.size(1) if sample.x is not None else 1,
        "num_node_types": int(sample.node_type.max().item() + 1)
        if len(dataset) == 1
        else int(max(d.node_type.max().item() for d in dataset) + 1),
        "num_edge_types": int(sample.edge_type.max().item() + 1)
        if len(dataset) == 1
        else int(max(d.edge_type.max().item() for d in dataset) + 1),
        "num_classes": int(max(labels) + 1),
    }
    return train_loader, val_loader, test_loader, meta
