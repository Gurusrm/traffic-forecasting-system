import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(GCNBlock, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # x: (Batch, Nodes, Features) -> GCN expects (Total_Nodes, Features) if batching graphs
        # But here we often process a single graph with batched time-steps.
        # Handling (Batch, Nodes, F) for static graph:
        
        B, N, F_dim = x.shape
        x_reshaped = x.reshape(B * N, F_dim)
        
        # Replicate edge_index for the batch? 
        # Or more simply, loop over batch (inefficient) or use PyG Batch object.
        # For simplicity in this bespoke T-GCN, we'll iterate or assume fixed graph.
        
        # EFFICIENT APPROACH: 
        # GCNConv in PyG supports dense batching if using DenseGCNConv, 
        # or we treat the batch dimension as just more nodes in a disconnected giant graph.
        
        # Let's use the giant graph approach.
        # x is (B*N, F)
        
        # We need to repeat edge_index for each batch item.
        # edge_index: (2, E)
        # Shift indices by N for each batch item.
        
        edge_index_batch = []
        edge_weight_batch = []
        
        # NOTE: Doing this repeat every forward pass is slow. 
        # Ideally pre-compute this or use PyTorch Geometric Temporal if available.
        # Since we are implementing from scratch, let's use a simpler approach:
        # Loop over batch (it's okay for 50 nodes and small batch sizes, but not optimal).
        
        # BETTER: Matrix multiplication formulation of GCN A*X*W
        # This is faster for fixed graph structure.
        # Support only fixed adjacency for now.
        
        pass 
        # We will actually implement a Dense GCN for speed on fixed graph 50 nodes.

class TGCNCell(nn.Module):
    """
    Temporal Graph Convolutional Cell
    Combines GRU and GCN.
    """
    def __init__(self, num_nodes, in_channels, hidden_channels, dropout=0.3):
        super(TGCNCell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        
        # GCN weights for GRU gates
        # Update Gate
        self.gcn_uz = GCNConv(in_channels, hidden_channels)
        self.gcn_hz = GCNConv(hidden_channels, hidden_channels)
        
        # Reset Gate
        self.gcn_ur = GCNConv(in_channels, hidden_channels)
        self.gcn_hr = GCNConv(hidden_channels, hidden_channels)
        
        # New Gate
        self.gcn_uh = GCNConv(in_channels, hidden_channels)
        self.gcn_hh = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, h, edge_index, edge_weight=None):
        # x: (Batch, Nodes, In_Channels)
        # h: (Batch, Nodes, Hidden_Channels)
        
        # We need to process the batch. 
        # Since PyG GCNConv expects (TotalNodes, C), we reshape.
        B, N, C = x.shape
        
        # Prepare Batch Graph Indices (Doing this efficiently)
        # Actually, for a fixed small graph, we can use the weight sharing property.
        # But GCNConv expects edge_index. 
        
        # Trick: Flatten Batch and Nodes -> (B*N, C)
        # Create a block-diagonal adjacency matrix (giant graph)
        # OR just loop if B is small (Example B=64, N=50). B*N = 3200 nodes.
        # It's fast enough to do giant graph.
        
        device = x.device
        
        # Construct batch edge_index
        # This should be passed in ideally to avoid re-creation
        # For this demo, we assume the caller handles the batching or we do it here.
        
        # Helper to apply GCN to (B, N, C) data with static edge_index
        def apply_gcn(gcn_layer, input_tensor):
            # input_tensor: (B, N, C)
            input_flat = input_tensor.view(-1, input_tensor.size(2)) # (B*N, C)
            
            # We need the batched edge_index. 
            # If we don't have it, we must compute it (expensive) or use a loop.
            # Let's try to assume input x is already (B*N, C) and edge_index is batched?
            # No, standard T-GCN usually takes (B, N, C).
            
            # Alternative: Dense GCN multiplication (A * X * W)
            # A: (N, N)
            # X: (B, N, C)
            # W: (C, Out)
            # A @ X @ W -> (B, N, Out)
            
            # Let's implement the logic of GCNConv manually for dense batching support
            # This avoids the edge_index batching complexity.
            
            weight = gcn_layer.lin.weight # (In, Out) typically, or (Out, In) Check PyG docs.
            # PyG GCNConv lin is Linear(in, out, bias=False)
            
            # Step 1: X @ W (Linear Transform)
            support = gcn_layer.lin(input_tensor) # (B, N, Out)
            
            # Step 2: A @ Support (Propagation)
            # We need the normalized adjacency matrix here.
            # This logic assumes `edge_index` -> Dense A
            
            return support # Placeholder if we don't have A
            
        # ... Reverting to standard GRU logic with PyG logic
        # For simplicity and correctness with PyG, let's just loop over the batch for now. 
        # It is readable and correct.
        
        h_new_list = []
        for b in range(B):
            x_b = x[b] # (N, In)
            h_b = h[b] # (N, Hidden)
            
            # Update gate
            z = torch.sigmoid(self.gcn_uz(x_b, edge_index, edge_weight) + self.gcn_hz(h_b, edge_index, edge_weight))
            
            # Reset gate
            r = torch.sigmoid(self.gcn_ur(x_b, edge_index, edge_weight) + self.gcn_hr(h_b, edge_index, edge_weight))
            
            # New Candidate
            h_hat = torch.tanh(self.gcn_uh(x_b, edge_index, edge_weight) + self.gcn_hh(r * h_b, edge_index, edge_weight))
            
            # Hidden State Update
            h_new = (1 - z) * h_b + z * h_hat
            h_new_list.append(h_new)
            
        return torch.stack(h_new_list, dim=0)
