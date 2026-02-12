import torch
import torch.nn as nn

class TemporalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # GRU input is the value from previous step (or 0)
        # Hidden state from encoder
        self.rnn = nn.GRU(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.num_layers = num_layers
        
    def forward(self, context, horizon=12):
        """
        context: (Batch, Nodes, Hidden) - Graph-enhanced representation
        horizon: int - steps to predict
        """
        B, N, H = context.shape
        
        # Reshape context to (num_layers, B*N, H) to use as initial hidden state
        # (1, B*N, H) -> (num_layers, B*N, H)
        h_0 = context.view(1, B * N, H).expand(self.num_layers, -1, -1).contiguous()
        
        # Inputs: We feed zeros for now (could be autoregressive with teacher forcing in future)
        # (B*N, Horizon, OutputDim)
        decoder_input = torch.zeros(B * N, horizon, self.output_dim, device=context.device)
        
        out, _ = self.rnn(decoder_input, h_0)
        # out: (B*N, horizon, H)
        
        preds = self.fc(out) # (B*N, horizon, Out)
        
        return preds.view(B, N, horizon, self.output_dim)
