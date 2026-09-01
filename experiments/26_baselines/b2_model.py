"""B2 - our implementation of GraphGeo's relation-specific encoder.

    m_agree_ij    = W_agree    . sigma(h_i (*) h_j)
    m_conflict_ij = W_conflict . sigma(h_i -  h_j)
    m_transfer_ij = W_transfer . h_j
    h_i^(l+1) = LayerNorm(ReLU(W_self h_i^(l) + SUM_r MEAN_j m_r_ij))

Two layers, hidden 32, dropout 0.5, DropEdge 0.2, weight decay 5e-4 - the same
capacity budget as ThothGNN v3, so capacity is not the confound.

Readout: SUM over agent nodes -> MLP -> one logit. NO ANCHOR. GraphGeo has no
such term and running it unanchored is the entire point of the comparison.
`--anchored` adds beta*logit(mal_share) and is the key row: if it matches
ThothGNN v3, our contribution is the anchor and not claim-level nodes.

torch is CPU-only here by design; the model is ~10k parameters and a GPU would
be slower than the kernel-launch overhead.
"""
from __future__ import annotations

import torch
import torch.nn as nn

REL = ("agree", "conflict", "transfer")


def _seg_mean(msg, dst, n):
    """Mean of messages arriving at each destination node."""
    out = torch.zeros(n, msg.shape[1], dtype=msg.dtype)
    cnt = torch.zeros(n, 1, dtype=msg.dtype)
    if msg.shape[0]:
        out.index_add_(0, dst, msg)
        cnt.index_add_(0, dst, torch.ones(msg.shape[0], 1, dtype=msg.dtype))
    return out / cnt.clamp(min=1.0)


class RelLayer(nn.Module):
    def __init__(self, din, dout, relations=REL, shared=False):
        super().__init__()
        self.relations, self.shared = tuple(relations), shared
        self.self_w = nn.Linear(din, dout)
        if shared:
            # B2-no-relation: ONE weight matrix for every edge type
            self.w = nn.ModuleDict({"shared": nn.Linear(din, dout)})
        else:
            self.w = nn.ModuleDict({r: nn.Linear(din, dout) for r in self.relations})
        self.norm = nn.LayerNorm(dout)

    def message(self, r, hi, hj):
        if r == "agree":
            z = torch.sigmoid(hi * hj)
        elif r == "conflict":
            z = torch.sigmoid(hi - hj)
        else:
            z = hj
        return self.w["shared" if self.shared else r](z)

    def forward(self, h, edges):
        agg = self.self_w(h)
        for r in self.relations:
            src, dst = edges[r]
            if src.numel() == 0:
                continue
            agg = agg + _seg_mean(self.message(r, h[dst], h[src]), dst, h.shape[0])
        return self.norm(torch.relu(agg))


class GraphGeo(nn.Module):
    def __init__(self, d_img, n_agents=6, d_emb=8, hidden=32, dropout=0.5,
                 relations=REL, use_agent_emb=True, shared_rel=False,
                 readout="sum", anchored=False):
        super().__init__()
        self.use_agent_emb, self.readout, self.anchored = use_agent_emb, readout, anchored
        self.emb = nn.Embedding(n_agents, d_emb) if use_agent_emb else None
        din = d_img + (d_emb if use_agent_emb else 0)
        self.l1 = RelLayer(din, hidden, relations, shared_rel)
        self.l2 = RelLayer(hidden, hidden, relations, shared_rel)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))
        self.beta = nn.Parameter(torch.zeros(1)) if anchored else None

    def forward(self, x, edges, batch, n_graphs, slot, anchor=None):
        """`batch` maps each node to its graph; `slot` to its agent index.
        Graphs are stacked block-diagonally so a whole fold is one forward pass."""
        if self.use_agent_emb:
            x = torch.cat([x, self.emb(slot)], dim=1)
        h = self.drop(self.l1(x, edges))
        h = self.l2(h, edges)
        g = torch.zeros(n_graphs, h.shape[1], dtype=h.dtype)
        g.index_add_(0, batch, h)
        if self.readout == "mean":
            c = torch.zeros(n_graphs, 1, dtype=h.dtype)
            c.index_add_(0, batch, torch.ones(h.shape[0], 1, dtype=h.dtype))
            g = g / c.clamp(min=1.0)
        s = self.head(g).squeeze(-1)
        if self.anchored and anchor is not None:
            s = s + self.beta * anchor
        return s


def collate(graphs, n_agents=6):
    """Stack graphs block-diagonally into one big disconnected graph."""
    xs, bat, slot, ys, anch = [], [], [], [], []
    src = {r: [] for r in REL}
    dst = {r: [] for r in REL}
    off = 0
    for gi, g in enumerate(graphs):
        xs.append(torch.as_tensor(g["x"]))
        bat += [gi] * n_agents
        slot += list(range(n_agents))
        ys.append(g["y"])
        m = min(max(g["mal_share"], 1e-3), 1 - 1e-3)
        anch.append(float(torch.log(torch.tensor(m / (1 - m)))))
        for r in REL:
            s_, d_ = g["edges"][r]
            if len(s_):
                src[r].append(torch.as_tensor(s_) + off)
                dst[r].append(torch.as_tensor(d_) + off)
        off += n_agents
    edges = {r: ((torch.cat(src[r]) if src[r] else torch.zeros(0, dtype=torch.long)),
                 (torch.cat(dst[r]) if dst[r] else torch.zeros(0, dtype=torch.long)))
             for r in REL}
    return {"x": torch.cat(xs), "edges": edges,
            "batch": torch.tensor(bat), "slot": torch.tensor(slot),
            "y": torch.tensor(ys, dtype=torch.float32),
            "anchor": torch.tensor(anch, dtype=torch.float32),
            "n": len(graphs)}


def drop_edge(edges, p, gen):
    """DropEdge: independently drop each edge with probability p."""
    if p <= 0:
        return edges
    out = {}
    for r, (src, dst) in edges.items():
        if src.numel() == 0:
            out[r] = (src, dst); continue
        keep = torch.rand(src.shape[0], generator=gen) >= p
        out[r] = (src[keep], dst[keep])
    return out
