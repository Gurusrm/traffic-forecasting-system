import torch
import numpy as np
from torch_geometric.data import Data

def load_adjacency_matrix(adj_path):
    """
    Load adjacency matrix from pkl or npz
    """
    import pickle
    try:
        with open(adj_path, 'rb') as f:
            adj_mx = pickle.load(f)
    except UnicodeDecodeError:
        with open(adj_path, 'rb') as f:
            adj_mx = pickle.load(f, encoding='latin1')
    return adj_mx

def correlation_adj_mx(data, threshold=0.5):
    """
    Construct adj matrix based on correlation of time series
    """
    # data: (T, N)
    corr = np.corrcoef(data.T)
    adj = np.where(np.abs(corr) > threshold, 1, 0)
    return adj

def get_graph_data(adj_mx, device='cpu'):
    """
    Convert adjacency matrix to PyG Data object
    """
    rows, cols = np.where(adj_mx > 0)
    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
    edge_attr = torch.tensor(adj_mx[rows, cols], dtype=torch.float, device=device)
    
    num_nodes = adj_mx.shape[0]
    
    # We don't have node features yet (x), just the graph structure
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
