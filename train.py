import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader
import networkx as nx

from src.models.tgcn import TGCN

# Configuration
DATA_DIR = 'data/processed'
CHECKPOINT_DIR = 'models/checkpoints'
EPOCHS = 5 # Reduced for demo speed
BATCH_SIZE = 32
LR = 0.001
INPUT_WINDOW = 4
OUTPUT_WINDOW = 1 # Predict 1 step ahead (5 mins) recursively for more
HIDDEN_DIM = 32

def load_data():
    print("Loading data...")
    speed = np.load(os.path.join(DATA_DIR, 'speed_data.npy')) # (T, N)
    adj = np.load(os.path.join(DATA_DIR, 'adj_matrix.npy'))
    
    # Normalize speed (0-1 approx or Z-score)
    mean = speed.mean()
    std = speed.std()
    speed_norm = (speed - mean) / std
    
    # Create windows
    # X: (Samples, Window, Nodes, 1)
    # Y: (Samples, Nodes, 1)
    
    X = []
    Y = []
    T, N = speed_norm.shape
    
    for i in range(T - INPUT_WINDOW - OUTPUT_WINDOW):
        X.append(speed_norm[i : i + INPUT_WINDOW, :].reshape(INPUT_WINDOW, N, 1))
        Y.append(speed_norm[i + INPUT_WINDOW, :].reshape(N, 1)) # Next step
        
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Convert to Tensor
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Prepare Graph
    adj_tensor = torch.FloatTensor(adj)
    edge_index = adj_tensor.nonzero().t().contiguous()
    edge_weight = adj_tensor[edge_index[0], edge_index[1]]
    
    return loader, edge_index, edge_weight, mean, std

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loader, edge_index, edge_weight, mean, std = load_data()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    num_nodes = edge_index.max().item() + 1
    
    model = TGCN(
        num_nodes=num_nodes,
        in_channels=1,
        hidden_channels=HIDDEN_DIM,
        output_dim=1,
        input_window=INPUT_WINDOW
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    model.train()
    
    
    # Setup Logging
    log_file = os.path.join(CHECKPOINT_DIR, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"Training started at {time.ctime()}\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
        else:
            f.write("WARNING: Training running on CPU.\n")
        f.write("-" * 30 + "\n")

    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_Y in loader:
            batch_X = batch_X.to(device) # (B, T, N, F)
            batch_Y = batch_Y.to(device) # (B, N, 1)
            
            optimizer.zero_grad()
            
            with autocast():
                pred = model(batch_X, edge_index, edge_weight)
                loss = criterion(pred, batch_Y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        msg = f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {time.time()-start_time:.1f}s"
        print(msg, flush=True)
        with open(log_file, 'a') as f:
            f.write(msg + "\n")
            
    # Save Model
    save_path = os.path.join(CHECKPOINT_DIR, 'tgcn_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Save Metadata
    np.save(os.path.join(CHECKPOINT_DIR, 'scaler_mean.npy'), mean)
    np.save(os.path.join(CHECKPOINT_DIR, 'scaler_std.npy'), std)

if __name__ == "__main__":
    train()
