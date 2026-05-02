import torch
import torch.nn as nn
from .temporal_encoder import TemporalEncoder
from .graph_layers import GraphLayer
from .temporal_decoder import TemporalDecoder

class STGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        model_cfg = config['model']
        data_cfg = config['data']
        
        self.encoder = TemporalEncoder(
            input_dim=model_cfg['input_dim'],
            hidden_dim=model_cfg['hidden_dim'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout']
        )
        
        self.graph_layer = GraphLayer(
            in_channels=model_cfg['hidden_dim'],
            out_channels=model_cfg['hidden_dim'],
            heads=model_cfg['heads'],
            dropout=model_cfg['dropout']
        )
        
        self.decoder = TemporalDecoder(
            input_dim=model_cfg['hidden_dim'], 
            hidden_dim=model_cfg['hidden_dim'],
            output_dim=model_cfg['output_dim'],
            num_layers=model_cfg['num_layers'],
            dropout=model_cfg['dropout']
        )
        
        self.horizon = data_cfg['output_window']
        
    def forward(self, x, edge_index):
        # x: (B, T, N, F)
        
        # 1. Encode
        # (B, N, H)
        temporal_emb = self.encoder(x)
        
        # 2. Graph Message Passing
        # (B, N, H)
        spatial_emb = self.graph_layer(temporal_emb, edge_index)
        
        # Residual connection
        context = spatial_emb + temporal_emb
        
        # 3. Decode
        # (B, N, T_out, F_out)
        predictions = self.decoder(context, horizon=self.horizon)
        
        # Return predictions in shape (B, T_out, N, F_out) to match target mostly?
        # Dataset returns (output_window, num_nodes, 1) usually.
        # Our decoder returns (B, N, T, 1).
        # Let's permute to (B, T, N, 1)
        return predictions.permute(0, 2, 1, 3)
