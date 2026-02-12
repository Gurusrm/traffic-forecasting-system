from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import yaml
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.models.st_gnn import STGNN
from src.data.graph_builder import load_adjacency_matrix, get_graph_data

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import random

# ... (Previous code)

app = FastAPI(title="Traffic Forecasting API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
edge_index = None
device = None
config = None
test_dataset = None
sensor_locations = None

# ... (Load modtest_dataset = None

class PredictionRequest(BaseModel):
    input_data: list 

@app.on_event("startup")
def load_model():
    global model, edge_index, device, config, test_dataset
    
    # ... (Existing loading logic) ...
    # Copy previous loading logic here but ensure we load test dataset too
    
    # Try multiple convenient paths for config
    possible_paths = ["config/default_config.yaml", "../../config/default_config.yaml"]
    config_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not config_path:
        print("Config file not found.", flush=True)
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() and config['training']['gpu_id'] is not None else 'cpu')
    
    # Load Graph
    adj_path = os.path.join(config['data']['raw_data_path'], "adj_mx.pkl")
    if not os.path.exists(adj_path):
        adj_path = os.path.join("../../", config['data']['raw_data_path'], "adj_mx.pkl")
        
    if os.path.exists(adj_path):
        from src.data.graph_builder import load_adjacency_matrix, get_graph_data
        adj_mx = load_adjacency_matrix(adj_path)
        graph_data = get_graph_data(adj_mx, device=device)
        edge_index = graph_data.edge_index
    else:
        print(f"Warning: Adjacency matrix not found at {adj_path}")

    # Init Model
    from src.models.st_gnn import STGNN
    model = STGNN(config)
    
    # Checkpoint
    checkpoint_path = config['inference']['model_path']
    if not os.path.exists(checkpoint_path):
         checkpoint_path = os.path.join("../../", checkpoint_path)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
    model.to(device)
    model.eval()
    
    # Load Test Data for Samples
    try:
        from src.data.preprocessing import generate_train_val_test
        data_path = os.path.join(config['data']['raw_data_path'], f"{config['data']['dataset_name'].lower()}.npz")
        if not os.path.exists(data_path):
             data_path = os.path.join("../../", config['data']['raw_data_path'], f"{config['data']['dataset_name'].lower()}.npz")
             
        if os.path.exists(data_path):
            data = np.load(data_path)['data']
            _, _, test_data = generate_train_val_test(data, 
                                                    train_ratio=config['data']['train_ratio'], 
                                                    val_ratio=config['data']['val_ratio'])
            test_dataset = test_data # Shape (T, N, F)
            print(f"Loaded test data: {test_dataset.shape}")
    except Exception as e:
        print(f"Failed to load test data: {e}")

    # Load Sensor Locations
    global sensor_locations
    sensor_locations = []
    try:
        import pandas as pd
        loc_path = os.path.join(config['data']['raw_data_path'], "graph_sensor_locations.csv")
        if not os.path.exists(loc_path):
            loc_path = os.path.join("../../", config['data']['raw_data_path'], "graph_sensor_locations.csv")
        
        if os.path.exists(loc_path):
            print(f"Loading locations from: {os.path.abspath(loc_path)}")
            df = pd.read_csv(loc_path)
            print(f"DataFrame loaded. Head:\n{df.head()}")
            
            # The CSV from DCRNN repo has 'index', 'sensor_id', 'latitude', 'longitude'
            # We trust 'index' corresponds to the graph node index 0..N
            if 'index' in df.columns:
                df = df.sort_values('index')
            
            ordered_locations = []
            for _, row in df.iterrows():
                ordered_locations.append({
                    "id": str(row['sensor_id']),
                    "lat": float(row['latitude']),
                    "lng": float(row['longitude'])
                })
            
            sensor_locations = ordered_locations
            print(f"Loaded {len(sensor_locations)} sensor locations from CSV.")
            
    except Exception as e:
        print(f"Failed to load locations: {e}")

@app.get("/locations")
def get_locations():
    """Return list of sensor locations ordered by graph node index"""
    global sensor_locations
    if not sensor_locations:
        return []
    return sensor_locations

@app.get("/sample")
def get_sample():
    """Get a random sample from test set"""
    global test_dataset, config
    if test_dataset is None:
        raise HTTPException(status_code=404, detail="Test data not loaded")
        
    # Pick a random start index
    # validation/test data is continuous, so we just pick a window
    # We need input_window + output_window
    total_len = len(test_dataset)
    in_win = config['data']['input_window']
    out_win = config['data']['output_window']
    
    idx = random.randint(0, total_len - in_win - out_win - 1)
    
    x = test_dataset[idx : idx + in_win]
    y = test_dataset[idx + in_win : idx + in_win + out_win]
    
    return {
        "input": x.tolist(),
        "target": y.tolist(),
        "start_index": idx
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    global model, edge_index, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        data = np.array(request.input_data, dtype=np.float32)
        tensor_data = torch.FloatTensor(data).to(device)
        
        # Add batch dim if missing
        if tensor_data.dim() == 3:
            tensor_data = tensor_data.unsqueeze(0)
            
        with torch.no_grad():
            preds = model(tensor_data, edge_index)
            
        # preds: (B, T, N, F)
        return {"prediction": preds.cpu().tolist()}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Mount frontend
app.mount("/", StaticFiles(directory="src/frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
