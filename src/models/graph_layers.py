import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv

class GraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0, gnn_type='GAT'):
        super().__init__()
        self.gnn_type = gnn_type
        
        if gnn_type == 'GAT':
            self.conv = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout, concat=True)
            self.out_channels = out_channels # Assumes out_channels is divisible by heads * (out/heads)
        elif gnn_type == 'GCN':
            self.conv = GCNConv(in_channels, out_channels)
            self.out_channels = out_channels
        else:
            raise ValueError(f"Unknown GNN type {gnn_type}")
            
    def forward(self, x, edge_index):
        """
        x: (Batch, Nodes, In_Features)
        edge_index: (2, E)
        """
        B, N, C = x.shape
        
        # Flatten batch for PyG
        x_reshaped = x.view(B * N, C)
        
        # We need to replicate edge_index for the batch or treat as one large graph
        # For efficiency, if the graph is the same for all batch items, we can extend edge_index
        # offset = torch.arange(B, device=x.device).view(-1, 1, 1) * N
        # This is expensive to do every forward pass.
        # Alternatively, we can pass a batched_edge_index if precomputed.
        # OR, we can just loop (slow)
        # OR, we rely on the caller to provide the correct `edge_index` for B*N nodes.
        
        # For now, let's assume the caller provides a batched edge_index OR we compute it here once if B is constant?
        # A simple robust way if we don't have pre-batched edge_index is to just process items? No that's too slow.
        
        # Let's assume edge_index is for a single graph (Nodes nodes).
        # We construct the block diagonal edge_index.
        # (This should ideally be done in the dataloader or model init if batch size is fixed, 
        # but batch size might vary (last batch)).
        
        # Hack for now: Construct batched edge_index on the fly.
        # In production this should be cached.
        
        edge_index_batched = self._batch_edge_index(edge_index, B, N)
        
        out = self.conv(x_reshaped, edge_index_batched)
        
        return out.view(B, N, -1)

    def _batch_edge_index(self, edge_index, batch_size, num_nodes):
        # edge_index: (2, E)
        # We want to repeat it B times with offsets
        # returns (2, E * B)
        
        num_edges = edge_index.shape[1]
        
        # (B, 1)
        offsets = torch.arange(batch_size, device=edge_index.device).view(-1, 1) * num_nodes
        
        # (1, 2, E)
        edges = edge_index.unsqueeze(0)
        
        # (B, 2, E) = (1, 2, E) + (B, 1, 1) [broadcasting (B, 1) to (B, 2)]
        # We need to add offset to both row and col indices
        offsets = offsets.unsqueeze(1) # (B, 1, 1)
        
        batched_edges = edges + offsets
        
        # Flatten to (2, B*E)
        batched_edges = batched_edges.permute(1, 0, 2).reshape(2, -1)
        
        return batched_edges
