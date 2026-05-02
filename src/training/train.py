import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.data.dataset import TrafficDataset
from src.data.graph_builder import load_adjacency_matrix, get_graph_data
from src.models.st_gnn import STGNN
from src.training.trainer import Trainer
from src.utils.logger import get_logger

def main(config_path):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    log_dir = config['training']['log_dir']
    logger = get_logger(log_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['gpu_id'] is not None else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load Data
    # Check if we need to generate dummy data if raw doesn't exist?
    raw_path = os.path.join(config['data']['raw_data_path'], f"{config['data']['dataset_name'].lower()}.npz")
    adj_path = os.path.join(config['data']['raw_data_path'], "adj_mx.pkl")
    
    if not os.path.exists(raw_path):
        logger.warning(f"Data not found at {raw_path}. Please run scripts/download_data.py to generate dummy data.")
        return

    logger.info(f"Loading data from {raw_path}")
    try:
        data = np.load(raw_path)['data']
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Split
    # data: (Samples, Nodes, Features)
    num_samples = len(data)
    train_len = int(num_samples * config['data']['train_ratio'])
    val_len = int(num_samples * config['data']['val_ratio'])
    
    train_data = data[:train_len]
    val_data = data[train_len : train_len + val_len]
    
    # Create Datasets
    input_window = config['data']['input_window']
    output_window = config['data']['output_window']
    
    train_dataset = TrafficDataset(train_data, input_window, output_window)
    val_dataset = TrafficDataset(val_data, input_window, output_window)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    # Load Graph
    logger.info(f"Loading graph from {adj_path}")
    if not os.path.exists(adj_path):
        logger.error(f"Adjacency matrix not found at {adj_path}")
        return
        
    adj_mx = load_adjacency_matrix(adj_path)
    graph_data = get_graph_data(adj_mx, device=device)
    edge_index = graph_data.edge_index
    
    # Init Model
    model = STGNN(config)
    
    # Trainer
    trainer = Trainer(model, config, device, logger)
    
    # Train
    trainer.train(train_loader, val_loader, edge_index, config['training']['epochs'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    args = parser.parse_args()
    
    main(args.config)
