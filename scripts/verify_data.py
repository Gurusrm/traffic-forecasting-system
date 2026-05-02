import pandas as pd
import pickle
import numpy as np
import os

def verify_data():
    h5_path = "data/raw/metr-la.h5"
    adj_path = "data/raw/adj_mx.pkl"
    
    print(f"Verifying {h5_path}...")
    try:
        df = pd.read_hdf(h5_path)
        print(f"Shape: {df.shape}")
        print(f"Time range: {df.index.min()} to {df.index.max()}")
        expected_shape = (34272, 207)
        if df.shape == expected_shape:
            print("SUCCESS: Shape matches expected METR-LA dimensions.")
        else:
            print(f"WARNING: Shape mismatch. Expected {expected_shape}, got {df.shape}")
            
    except Exception as e:
        print(f"FAILED to read h5: {e}")

    print(f"\nVerifying {adj_path}...")
    try:
        with open(adj_path, 'rb') as f:
            adj_mx = pickle.load(f)
        
        if isinstance(adj_mx, tuple):
             sensor_ids, sensor_id_to_ind, adj_mat = adj_mx
             print(f"Structure: Tuple (ids, map, mat)")
             print(f"Adjacency Matrix Shape: {adj_mat.shape}")
        elif isinstance(adj_mx, np.ndarray):
             print(f"Structure: Numpy Array")
             print(f"Shape: {adj_mx.shape}")
        else:
             print(f"Unexpected adj structure type: {type(adj_mx)}")

    except Exception as e:
        print(f"FAILED to read pickle: {e}")

if __name__ == "__main__":
    verify_data()
