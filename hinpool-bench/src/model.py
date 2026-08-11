import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class RGCNBaseline(nn.Module):
    """Two-layer RGCN + global mean pool. Baseline before HINPool."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_relations: int,
        num_classes: int,
        num_bases: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden, hidden, num_relations, num_bases=num_bases)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, data):
        x = data.x
        x = F.relu(self.conv1(x, data.edge_index, data.edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index, data.edge_type))
        x = global_mean_pool(x, data.batch)
        return self.head(x)


# ---------------------------------------------------------------------------
# HINPool components
# ---------------------------------------------------------------------------

class TAS(nn.Module):
    """Type-Aware Selector: per-type top-K selection with gated embeddings.

    Operates on a SINGLE graph's node embeddings.
    """

    def __init__(self, in_dim: int, num_types: int, pool_ratio: float = 0.9):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.num_types = num_types
        self.scorers = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_types)])

    def forward(self, x: torch.Tensor, node_type: torch.Tensor):
        """
        Returns:
            x_kept: [K, d] gated embeddings of kept nodes
            kept_idx: [K] sorted indices into x
        """
        n = x.size(0)
        scores = torch.zeros(n, device=x.device)

        for t, scorer in enumerate(self.scorers):
            mask = (node_type == t).nonzero(as_tuple=True)[0]
            if len(mask) > 0:
                scores[mask] = torch.sigmoid(scorer(x[mask]).squeeze(-1))

        parts = []
        for t in range(self.num_types):
            mask = (node_type == t).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                continue
            k = max(1, int(self.pool_ratio * len(mask)))
            _, topk_local = scores[mask].topk(k, sorted=False)
            parts.append(mask[topk_local])

        kept_idx = torch.cat(parts).sort().values if parts else torch.arange(n, device=x.device)
        x_kept = x[kept_idx] * scores[kept_idx].unsqueeze(-1)
        return x_kept, kept_idx


class RA(nn.Module):
    """Readout Aggregator: per-type mean + global readout → concat."""

    def __init__(self, in_dim: int, num_types: int, attn_pool: bool = False):
        super().__init__()
        self.num_types = num_types
        self.attn_pool = attn_pool
        self.out_dim = in_dim * (num_types + 1)
        if attn_pool:
            self.attn_scorer = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        d = x.size(1)
        type_means = []
        for t in range(self.num_types):
            mask = (node_type == t)
            type_means.append(x[mask].mean(dim=0) if mask.any() else torch.zeros(d, device=x.device))
        h_type = torch.cat(type_means)

        if self.attn_pool:
            attn_w = torch.softmax(self.attn_scorer(x).squeeze(-1), dim=0)
            h_global = (attn_w.unsqueeze(-1) * x).sum(dim=0)
        else:
            h_global = x.mean(dim=0)

        return torch.cat([h_type, h_global])


# ---------------------------------------------------------------------------
# HINPool full model
# ---------------------------------------------------------------------------

class HINPool(nn.Module):
    """
    HINPool (AAAI 2026): stacked THeGP layers with cross-layer fusion.

    Efficient implementation: RGCN runs over the full batch each layer;
    only TAS/RA/adjacency-rebuild uses a per-graph Python loop (cheap).

    attn_pool=True: attention-weighted global readout (ThothGNN variant).
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_relations: int,
        num_types: int,
        num_classes: int,
        num_layers: int = 3,
        pool_ratio: float = 0.9,
        num_bases: int = 8,
        dropout: float = 0.3,
        attn_pool: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        dims = [in_dim] + [hidden] * num_layers
        self.convs = nn.ModuleList([
            RGCNConv(dims[i], dims[i + 1], num_relations, num_bases=num_bases)
            for i in range(num_layers)
        ])
        self.tas_list = nn.ModuleList([
            TAS(dims[i + 1], num_types, pool_ratio)
            for i in range(num_layers)
        ])
        self.ra_list = nn.ModuleList([
            RA(dims[i + 1], num_types, attn_pool=attn_pool)
            for i in range(num_layers)
        ])

        ra_dim = hidden * (num_types + 1)
        fused_dim = ra_dim * num_layers  # cross-layer fusion by concatenation

        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    @staticmethod
    def _ptr_from_batch(batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        counts = torch.bincount(batch, minlength=num_graphs)
        ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=batch.device)
        ptr[1:] = counts.cumsum(0)
        return ptr

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type
        node_type = data.node_type
        batch = data.batch
        num_graphs = data.num_graphs

        all_readouts = []  # [num_layers] tensors of shape [B, ra_dim]

        for conv, tas, ra in zip(self.convs, self.tas_list, self.ra_list):
            # ---- RGCN over full batch: single fast batched op ----
            x = F.relu(conv(x, edge_index, edge_type))

            # ---- per-graph TAS + RA + adjacency rebuild (lightweight) ----
            ptr = self._ptr_from_batch(batch, num_graphs)

            new_x, new_nt, new_ei, new_et, new_batch = [], [], [], [], []
            layer_readouts = []
            new_offset = 0

            for g in range(num_graphs):
                s, e = ptr[g].item(), ptr[g + 1].item()
                x_g = x[s:e]
                nt_g = node_type[s:e]

                x_g_kept, kept_idx = tas(x_g, nt_g)
                nt_g_kept = nt_g[kept_idx]
                layer_readouts.append(ra(x_g_kept, nt_g_kept))

                # remap adjacency to retained nodes
                n_g = e - s
                node_map = torch.full((n_g,), -1, dtype=torch.long, device=x.device)
                node_map[kept_idx] = torch.arange(len(kept_idx), device=x.device)

                emask = (edge_index[0] >= s) & (edge_index[0] < e)
                src_g = edge_index[0, emask] - s
                dst_g = edge_index[1, emask] - s
                et_g = edge_type[emask]

                keep = (node_map[src_g] >= 0) & (node_map[dst_g] >= 0)
                ei_new = torch.stack([node_map[src_g[keep]], node_map[dst_g[keep]]]) + new_offset

                new_x.append(x_g_kept)
                new_nt.append(nt_g_kept)
                new_ei.append(ei_new)
                new_et.append(et_g[keep])
                new_batch.append(torch.full((len(kept_idx),), g, dtype=torch.long, device=x.device))
                new_offset += len(kept_idx)

            x = torch.cat(new_x)
            node_type = torch.cat(new_nt)
            edge_index = (
                torch.cat(new_ei, dim=1)
                if any(t.numel() > 0 for t in new_ei)
                else torch.zeros(2, 0, dtype=torch.long, device=x.device)
            )
            edge_type = torch.cat(new_et)
            batch = torch.cat(new_batch)

            all_readouts.append(torch.stack(layer_readouts))  # [B, ra_dim]

        fused = torch.cat(all_readouts, dim=1)  # [B, num_layers * ra_dim]
        return self.head(fused)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, meta: dict, hparams: dict) -> nn.Module:
    if model_name == "rgcn":
        return RGCNBaseline(
            in_dim=meta["num_node_features"],
            hidden=hparams.get("hidden", 64),
            num_relations=meta["num_edge_types"],
            num_classes=meta["num_classes"],
            num_bases=hparams.get("num_bases", 8),
            dropout=hparams.get("dropout", 0.3),
        )
    if model_name == "hinpool":
        return HINPool(
            in_dim=meta["num_node_features"],
            hidden=hparams.get("hidden", 64),
            num_relations=meta["num_edge_types"],
            num_types=meta["num_node_types"],
            num_classes=meta["num_classes"],
            num_layers=hparams.get("num_layers", 3),
            pool_ratio=hparams.get("pool_ratio", 0.9),
            num_bases=hparams.get("num_bases", 8),
            dropout=hparams.get("dropout", 0.3),
            attn_pool=hparams.get("attn_pool", False),
        )
    raise ValueError(f"Unknown model: {model_name!r}")
