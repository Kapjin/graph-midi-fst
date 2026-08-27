"""R-GCN variant of train_graph_muq_fst_logging.py.

All non-graph behavior is shared with the GAT entry point. The four audio
relations become integer edge types consumed by PyG's RGCNConv.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import RGCNConv
except ImportError as exc:
    raise SystemExit(
        "The R-GCN variant requires PyTorch Geometric. Activate .venv-midi and "
        "run: pip install torch_geometric"
    ) from exc

import train_graph_muq_fst_logging as base


class MultiRelationRGCNEncoder(nn.Module):
    """A PyG R-GCN whose edge types are the four explicit audio relations."""

    def __init__(
        self,
        input_dim=base.MUQ_DIM,
        relation_dim=256,
        num_heads=4,
        dropout=0.1,
    ) -> None:
        super().__init__()
        del num_heads  # Retained in the shared CLI/checkpoint schema.
        self.conv = RGCNConv(
            input_dim,
            relation_dim,
            num_relations=len(base.GRAPH_RELATIONS),
            aggr="mean",
        )
        self.fusion = nn.Linear(relation_dim, input_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, embeddings, adjacency, padding_mask):
        if adjacency.ndim != 4 or adjacency.shape[1] != len(base.GRAPH_RELATIONS):
            raise ValueError("Adjacency must have shape [B, 4, T, T]")
        outputs = []
        for node_features, relations, mask in zip(
            embeddings, adjacency, padding_mask
        ):
            node_count = int((~mask).sum().item())
            edge_parts = []
            type_parts = []
            for relation_id in range(len(base.GRAPH_RELATIONS)):
                weights = relations[relation_id, :node_count, :node_count]
                target, source = (weights > 0).nonzero(as_tuple=True)
                edge_parts.append(torch.stack((source, target), dim=0))
                type_parts.append(
                    torch.full_like(source, relation_id, dtype=torch.long)
                )
            edge_index = torch.cat(edge_parts, dim=1)
            edge_type = torch.cat(type_parts, dim=0)
            propagated = self.conv(
                node_features[:node_count], edge_index, edge_type
            )
            padded = node_features.new_zeros(
                (node_features.shape[0], propagated.shape[-1])
            )
            padded[:node_count] = propagated
            outputs.append(padded)
        message = torch.stack(outputs, dim=0)
        message = self.dropout(self.activation(self.fusion(message)))
        output = self.layer_norm(embeddings + message)
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


base.MultiRelationGATEncoder = MultiRelationRGCNEncoder
base.GRAPH_ARCHITECTURE = "relational-gcn"
base.GRAPH_DISPLAY_NAME = "R-GCN"
base.MUQ_FST_BUILD = "2026-08-27-graph-muq-rgcn-audio-r1"

if __name__ == "__main__":
    base.main()
