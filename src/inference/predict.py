import argparse
import yaml
import torch
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.models.st_gnn import STGNN
from src.data.graph_builder import load_adjacency_matrix, get_graph_data

def predict(config_path, checkpoint_path, output_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['gpu_id'] is not None else 'cpu')
    
    # Load Graph
    adj_path = os.path.join(config['data']['raw_data_path'], "adj_mx.pkl")
    adj_mx = load_adjacency_matrix(adj_path)
    graph_data = get_graph_data(adj_mx, device=device)
    edge_index = graph_data.edge_index
    
    # Init Model
    model = STGNN(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load Dataset
    from src.data.dataset import TrafficDataset
    from src.data.preprocessing import generate_train_val_test
    
    # Load raw data
    data_path = os.path.join(config['data']['raw_data_path'], f"{config['data']['dataset_name'].lower()}.npz")
    if os.path.exists(data_path):
        data = np.load(data_path)['data']
    else:
        # Fallback for dummy
        data = np.random.randn(1000, 207, 2)
        
    _, _, test_data = generate_train_val_test(data, 
                                            train_ratio=config['data']['train_ratio'], 
                                            val_ratio=config['data']['val_ratio'])
                                            
    dataset = TrafficDataset(test_data, 
                           input_window=config['data']['input_window'],
                           output_window=config['data']['output_window'])
                           
    # Get a sample batch (e.g., first 64 samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    x, y = next(iter(loader))
    
    x = x.to(device)
    y = y.to(device) # Targets
    
    print("Running inference...")
    with torch.no_grad():
        preds = model(x, edge_index)
        
    print(f"Prediction shape: {preds.shape}")
    print(f"Target shape: {y.shape}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, prediction=preds.cpu().numpy(), target=y.cpu().numpy())
    print(f"Saved prediction and target to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pth")
    parser.add_argument("--output", type=str, default="outputs/predictions/pred.npz")
    args = parser.parse_args()
    
    predict(args.config, args.checkpoint, args.output)
