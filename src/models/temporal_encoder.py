import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0, rnn_type='GRU'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            
    def forward(self, x):
        """
        x: (Batch, Seq, Nodes, Features)
        Returns: (Batch, Nodes, Hidden) - The last hidden state for each node
        """
        B, T, N, F = x.shape
        
        # Reshape to process each node's sequence independently
        # (Batch * Nodes, Seq, Features)
        x_reshaped = x.view(B * N, T, F)
        
        output, h_n = self.rnn(x_reshaped)
        
        # We take the output of the last time step
        # output: (B*N, T, H)
        last_step = output[:, -1, :] # (B*N, H)
        
        # Reshape back to (Batch, Nodes, Hidden)
        return last_step.view(B, N, self.hidden_dim)
