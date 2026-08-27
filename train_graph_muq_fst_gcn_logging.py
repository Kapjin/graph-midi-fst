"""GCN variant of train_graph_muq_fst_logging.py.

All audio preprocessing, MuQ, FST, training, inference, and logging behavior is
shared with the GAT entry point. Only the graph encoder uses PyG's GCNConv.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
except ImportError as exc:
    raise SystemExit(
        "The GCN variant requires PyTorch Geometric. Activate .venv-midi and "
        "run: pip install torch_geometric"
    ) from exc

import train_graph_muq_fst_logging as base


class RelationGCNBranch(nn.Module):
    """One weighted PyG GCN branch for a single audio relation."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = GCNConv(
            input_dim, output_dim, add_self_loops=False, normalize=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, adjacency, padding_mask):
        outputs = []
        for node_features, weights, mask in zip(
            embeddings, adjacency, padding_mask
        ):
            node_count = int((~mask).sum().item())
            valid_weights = weights[:node_count, :node_count].clamp_min(0.0)
            target, source = (valid_weights > 0).nonzero(as_tuple=True)
            edge_index = torch.stack((source, target), dim=0)
            edge_weight = valid_weights[target, source]
            propagated = self.conv(
                node_features[:node_count], edge_index, edge_weight
            )
            padded = node_features.new_zeros(
                (node_features.shape[0], propagated.shape[-1])
            )
            padded[:node_count] = self.dropout(propagated)
            outputs.append(padded)
        return torch.stack(outputs, dim=0)


class MultiRelationGCNEncoder(nn.Module):
    """Four weighted GCN branches followed by the original residual fusion."""

    def __init__(
        self,
        input_dim=base.MUQ_DIM,
        relation_dim=256,
        num_heads=4,
        dropout=0.1,
    ) -> None:
        super().__init__()
        del num_heads  # Retained in the shared CLI/checkpoint schema.
        self.branches = nn.ModuleList(
            RelationGCNBranch(input_dim, relation_dim, dropout)
            for _ in base.GRAPH_RELATIONS
        )
        self.fusion = nn.Linear(
            relation_dim * len(base.GRAPH_RELATIONS), input_dim
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, embeddings, adjacency, padding_mask):
        if adjacency.ndim != 4 or adjacency.shape[1] != len(base.GRAPH_RELATIONS):
            raise ValueError("Adjacency must have shape [B, 4, T, T]")
        relation_outputs = [
            branch(embeddings, adjacency[:, index], padding_mask)
            for index, branch in enumerate(self.branches)
        ]
        message = torch.cat(relation_outputs, dim=-1)
        message = self.dropout(self.activation(self.fusion(message)))
        output = self.layer_norm(embeddings + message)
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


base.MultiRelationGATEncoder = MultiRelationGCNEncoder
base.GRAPH_ARCHITECTURE = "relation-specific-gcn"
base.GRAPH_DISPLAY_NAME = "relation-specific GCN"
base.MUQ_FST_BUILD = "2026-08-27-graph-muq-gcn-audio-r1"

if __name__ == "__main__":
    base.main()
