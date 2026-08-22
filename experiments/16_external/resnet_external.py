"""ResNet-18: retrain on BreastMNIST, then apply unchanged to BUS-BRA.

The original baseline (test AUC 0.9442) saved predictions and metrics but no
weights, so the supervised arm of E1 has to be retrained before it can be
transferred. Same script settings as that run -- ImageNet-pretrained ResNet-18,
Adam 1e-3, 100 epochs, milestones 50/75, batch 128, seed 0, model selected by
best validation AUC. Reproduction is checked against 0.9442 before the external
number is believed; a retrain that misses in-domain is not a valid transfer test.

Data comes from the local npz rather than the medmnist downloader so that the
supervised arm sees byte-identical images to the probe arm.

The external pass applies the frozen 0.5 decision threshold unchanged, which is
what deploying a supervised model actually means.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_BM = _ROOT / "breastMnist/data/breast/images_224"
MALIGNANT = 0                      # npz convention; y=1 means malignant


def to_tensor(imgs):
    x = torch.from_numpy(imgs).float().div_(255.0).unsqueeze(1).repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def load_bm(split):
    z = np.load(_BM / f"{split}.npz")
    y = torch.from_numpy((z["labels"][:, 0] == MALIGNANT).astype("int64"))
    return to_tensor(z["imgs"]), y


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); o = a.argsort(kind="mergesort")
    r = np.empty(len(a)); sa = a[o]; i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def bacc(y, p, t=0.5):
    pred = (p >= t)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    return 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))


def predict(model, x, dev, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(torch.softmax(model(x[i:i + bs].to(dev)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    import random
    from torchvision.models import ResNet18_Weights, resnet18
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = _HERE / "results/resnet18_seed0.pt"

    xtr, ytr = load_bm("train"); xva, yva = load_bm("val"); xte, yte = load_bm("test")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(dev)

    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=dev))
        print(f"loaded existing checkpoint {ck.name}")
    else:
        g = torch.Generator().manual_seed(seed)
        tl = DataLoader(TensorDataset(xtr, ytr), batch_size=128, shuffle=True, generator=g)
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50, 75], gamma=0.1)
        best, best_state, best_ep, t0 = -1.0, None, -1, time.time()
        for ep in range(100):
            model.train()
            for xb, yb in tl:
                opt.zero_grad(); crit(model(xb.to(dev)), yb.to(dev)).backward(); opt.step()
            sch.step()
            v = auc(yva.numpy(), predict(model, xva, dev))
            if v > best:
                best, best_ep = v, ep
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            if (ep + 1) % 10 == 0:
                print(f"  epoch {ep+1:3d}/100 val_auc={v:.4f} (best {best:.4f} @ {best_ep})  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
        model.load_state_dict(best_state)
        torch.save(best_state, ck)
        print(f"trained in {(time.time()-t0)/60:.1f} min, best val AUC {best:.4f} @ epoch {best_ep}")

    pte = predict(model, xte, dev)
    ind_auc, ind_bacc = auc(yte.numpy(), pte), bacc(yte.numpy(), pte)
    print(f"\nin-domain BreastMNIST test: AUC {ind_auc:.4f}  bAcc {ind_bacc:.4f}"
          f"   (original run 0.9442 / 0.8672)")
    ok = abs(ind_auc - 0.9442) < 0.03
    print(f"reproduction {'OK' if ok else 'MISMATCH -- external transfer not valid'}")

    res = {"indomain": {"auc": float(ind_auc), "bacc": float(ind_bacc),
                        "original_auc": 0.9442, "reproduced": bool(ok)}}

    for tag, npz in (("busbra_pad2", "busbra_pad2_224.npz"),
                     ("busbra_pad15", "busbra_pad15_224.npz"),
                     ("breast", "breast_pad2_224.npz")):
        z = np.load(_ROOT / f"data/external/{npz}")
        y = (z["labels"][:, 0] == MALIGNANT).astype(int)
        p = predict(model, to_tensor(z["imgs"]), dev)
        r = {"n": int(len(y)), "auc": float(auc(y, p)), "bacc": float(bacc(y, p))}
        if "cases" in z.files:                       # case-level is the primary endpoint
            cs = z["cases"]
            u = np.unique(cs)
            pc = np.array([p[cs == c].mean() for c in u])
            yc = np.array([y[cs == c][0] for c in u])
            r.update({"n_cases": int(len(u)), "auc_case": float(auc(yc, pc)),
                      "bacc_case": float(bacc(yc, pc))})
        res[tag] = r
        extra = (f"  | case-level AUC {r['auc_case']:.4f} bAcc {r['bacc_case']:.4f}"
                 if "auc_case" in r else "")
        print(f"{tag:14s} n={r['n']:5d}  AUC {r['auc']:.4f}  bAcc {r['bacc']:.4f}{extra}")
        np.save(_HERE / f"results/resnet_scores_{tag}.npy", p)

    (_HERE / "results/resnet_external.json").write_text(json.dumps(res, indent=1))
    print(f"\ngeneralisation gap (in-domain -> BUS-BRA case-level): "
          f"{ind_auc - res['busbra_pad2']['auc_case']:+.4f} AUC")


if __name__ == "__main__":
    main()
