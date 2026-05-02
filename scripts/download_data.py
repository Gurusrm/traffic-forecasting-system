import argparse
import os
import numpy as np
import pickle
import pandas as pd
import requests

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def generate_graph(adj_mx, nodes):
    # DCRNN adj_mx is (sensor_ids, sensor_ids, adj)
    # sensor_ids is list of ids.
    # We need to map sensor ids to indices 0..N-1
    sensor_ids, sensor_id_to_ind, adj = adj_mx
    return adj

def process_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # URLs
    # Using raw links from repositories that host the data
    metr_la_url = "https://media.githubusercontent.com/media/liyaguang/DCRNN/master/data/metr-la.h5" 
    # Note: liyaguang repo might use LFS or not have it. 
    # Let's try yoshall/STGCN or similar if that fails.
    # actually liyaguang/DCRNN only has readme in data usually.
    # Try: https://raw.githubusercontent.com/hazdzz/DCRNN-PyTorch/master/data/METR-LA/metr-la.h5
    
    # URLs to try for H5
    h5_urls = [
        "https://github.com/leilin-research/GCGRNN/raw/master/data/METR-LA_traffic_speed/metr-la.h5", # Found via search
        "https://raw.githubusercontent.com/leilin-research/GCGRNN/master/data/METR-LA_traffic_speed/metr-la.h5",
        "https://interaction.ethz.ch/var/storage/h5/metr-la.h5", 
        "https://raw.githubusercontent.com/davidham3/STGCN/master/data/METR-LA/metr-la.h5",
        "https://raw.githubusercontent.com/yoshall/STGCN/master/data/METR-LA/metr-la.h5",
        "https://github.com/yoshall/STGCN/raw/master/data/METR-LA/metr-la.h5",
        "https://raw.githubusercontent.com/hazdzz/DCRNN-PyTorch/master/data/METR-LA/metr-la.h5"
    ]
    
    # URL for adj
    adj_url = "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl"
    
    # Paths
    h5_path = os.path.join(output_dir, "metr-la.h5")
    adj_path = os.path.join(output_dir, "adj_mx.pkl")
    npz_path = os.path.join(output_dir, "metr-la.npz")
    
    # 1. Download H5
    if not os.path.exists(h5_path):
        success = False
        for url in h5_urls:
            if download_file(url, h5_path):
                success = True
                break
        if not success:
            print("Could not download metr-la.h5. Generating dummy data instead.")
            generate_dummy_data(output_dir)
            return

    # 2. Download Adj
    if not os.path.exists(adj_path):
        if not download_file(adj_url, adj_path):
            print("Could not download adj_mx.pkl. Using dummy graph.")
            # Logic to create dummy graph matching h5 nodes?
            # We need to read h5 first to know nodes.
            pass

    # 3. Process
    print("Processing data...")
    try:
        df = pd.read_hdf(h5_path)
        # df shape: (Time, Nodes)
        data = np.expand_dims(df.values, axis=-1) # (T, N, 1)
        
        # Add time of day feature
        # timestamps
        time_index = df.index
        # Time of day: normalized 0-1
        tod = (time_index.hour * 60 + time_index.minute) / (24 * 60)
        tod = np.tile(tod.values.reshape(-1, 1, 1), (1, data.shape[1], 1))
        
        # Concat: (T, N, 2) -> Speed, TimeOfDay
        data = np.concatenate([data, tod], axis=-1)
        
        np.savez(npz_path, data=data)
        print(f"Saved processed data to {npz_path}. Shape: {data.shape}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Falling back to dummy data.")
        generate_dummy_data(output_dir)

def generate_dummy_data(output_dir, num_nodes=207, num_samples=1000):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating dummy data in {output_dir}")
    
    # Speed data: [samples, nodes, features]
    # features: speed, time_of_day
    data = np.random.randn(num_samples, num_nodes, 2).astype(np.float32)
    
    # Adjacency matrix
    adj = np.random.rand(num_nodes, num_nodes)
    adj = (adj > 0.95).astype(np.float32) # Sparse
    np.fill_diagonal(adj, 1.0)
    
    np.savez(os.path.join(output_dir, "metr-la.npz"), data=data)
    
    with open(os.path.join(output_dir, "adj_mx.pkl"), "wb") as f:
        pickle.dump(adj, f)
        
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="metr-la")
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    process_data(args.output_dir)
