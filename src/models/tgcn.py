import torch
import torch.nn as nn
from .layers import TGCNCell

class TGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, output_dim, input_window):
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.input_window = input_window
        self.hidden_channels = hidden_channels
        
        self.tgcn_cell = TGCNCell(num_nodes, in_channels, hidden_channels)
        
        # Predictor layer
        self.regressor = nn.Linear(hidden_channels, output_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        # x: (Batch, Input_Window, Nodes, Features)
        
        B, T, N, F = x.shape
        
        # Initialize hidden state
        h = torch.zeros(B, N, self.hidden_channels, device=x.device)
        
        # Process sequence
        for t in range(T):
            x_t = x[:, t, :, :] # (Batch, Nodes, Features)
            h = self.tgcn_cell(x_t, h, edge_index, edge_weight)
            
        # h is now the state after seeing T steps
        # We want to predict the next step (or sequence).
        # For simplicity, this model predicts the NEXT step for each node.
        
        prediction = self.regressor(h) # (Batch, Nodes, Output_Dim)
        
        return prediction
