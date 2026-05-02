import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import networkx as nx
import os
import sys
import random
import urllib.parse
from datetime import datetime, timedelta
import torch

# Optional heavy dependency
try:
    import torch
except Exception:
    torch = None

# Add src to path
sys.path.append(os.path.abspath('.'))

# Try project model & routing utilities, fallback to local stubs
try:
    from src.models.tgcn import TGCN
except Exception:
    TGCN = None

try:
    from src.utils.routing import build_graph_from_adj, calculate_route
except Exception:
    build_graph_from_adj = None
    calculate_route = None

print("DEBUG: App starting up...")

# Configuration
DATA_DIR = 'data/processed'
CHECKPOINT_DIR = 'models/checkpoints'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'tgcn_model.pth')
NUM_SENSORS = 50
HIDDEN_DIM = 32
INPUT_WINDOW = 4

st.set_page_config(layout="wide", page_title="UrbanFlow AI Command Center", initial_sidebar_state="expanded")

# Custom CSS for Command Center Aesthetic
st.markdown("""
<style>
    /* Dark Glassmorphism Theme */
    .stMetric { 
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(10px);
        padding: 15px; 
        border-radius: 15px; 
        border: 1px solid rgba(255,255,255,0.1); 
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetricLabel { color: #a8b2d1 !important; }
    .stMetricValue { color: #00c6ff !important; font-weight: 600; text-shadow: 0 0 10px rgba(0,198,255,0.5); }
    div[data-testid="stExpander"] { 
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(10px);
        border-radius: 15px; 
        border: 1px solid rgba(255,255,255,0.1); 
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        border: none;
        color: white;
        border-radius: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6);
        color: white;
    }
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_static_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    adj_path = os.path.join(DATA_DIR, 'adj_matrix.npy')
    dist_path = os.path.join(DATA_DIR, 'dist_matrix.npy')
    speed_path = os.path.join(DATA_DIR, 'speed_data.npy')

    # If files exist, load them
    if os.path.exists(adj_path) and os.path.exists(dist_path) and os.path.exists(speed_path):
        adj = np.load(adj_path)
        dist = np.load(dist_path)
        speed = np.load(speed_path)
        return adj, dist, speed

    # Otherwise generate synthetic data and save for reproducibility
    np.random.seed(42)
    positions = np.random.rand(NUM_SENSORS, 2)
    # compute euclidean distances scaled
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    
    # Adjust k to 4 to prevent messy spaghetti routing
    k = 4
    adj = np.zeros_like(dist)
    for i in range(NUM_SENSORS):
        idx = np.argsort(dist[i])[1:k+1]
        adj[i, idx] = 1
        adj[idx, i] = 1
    
    # Make adjacency symmetric properly
    adj = np.maximum(adj, adj.T)
    
    # speed data: times x nodes
    timesteps = 240
    
    # Make some nodes central (e.g., distance to center < 0.3)
    center = np.array([0.5, 0.5])
    dist_to_center = np.linalg.norm(positions - center, axis=-1)
    is_central = dist_to_center < 0.3
    
    # Central nodes have severe traffic jams (speed drops to 5-15 km/h)
    # Outer nodes are highways (speed 60-80 km/h)
    speed = np.zeros((timesteps, NUM_SENSORS))
    for i in range(NUM_SENSORS):
        # Phase shift based on X coordinate so traffic moves West to East over time
        phase_shift = positions[i, 0] * 4 * np.pi
        
        # Everyone has a fluctuating base speed, causing dynamic bottlenecks
        base = 45 + 35 * np.sin(np.linspace(0, 8 * np.pi, timesteps) + phase_shift)
        noise = np.random.randn(timesteps) * 5
        
        # Central nodes are generally worse, but not always
        if is_central[i]:
            base -= 20
            
        speed[:, i] = np.clip(base + noise, 5, 80)

    np.save(adj_path, adj)
    np.save(dist_path, dist)
    np.save(speed_path, speed)
    return adj, dist, speed

@st.cache_resource
def load_model():
    device = type("D", (), {"type": "cpu"})()  # simple device object with .type
    # If torch available and project model exists, attempt load
    if torch is not None and TGCN is not None and os.path.exists(MODEL_PATH):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            model = TGCN(NUM_SENSORS, HIDDEN_DIM) if TGCN is not None else None
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            if hasattr(model, 'to'):
                model.to(device)
            return model, device
        except Exception:
            pass

    # Fallback dummy model that returns small perturbation of last input
    class DummyModel:
        def __init__(self):
            pass
        def predict(self, x):  # x shape: (window, nodes)
            last = x[-1] if x.ndim == 2 else x.squeeze()[-1]
            return np.clip(last * (1 + np.random.randn(*last.shape) * 0.02), 1.0, 120.0)
        def __call__(self, x):
            return self.predict(x)
    return DummyModel(), device

def get_node_positions():
    """
    Returns fixed node positions for visualization (lat, lon).
    """
    np.random.seed(42)
    node_pos = np.random.rand(NUM_SENSORS, 2)
    lat_base, lon_base = 10.7905, 78.7047
    lat_spread, lon_spread = 0.06, 0.06
    positions = {}
    for i in range(NUM_SENSORS):
        lat = lat_base + (node_pos[i,0] - 0.5) * lat_spread
        lon = lon_base + (node_pos[i,1] - 0.5) * lon_spread
        positions[i] = (lat, lon)
    return positions

def calculate_path_cost(dist_matrix, path, speeds):
    """
    Calculates total travel time for a path given speeds per node (km/h numeric).
    Uses distances in arbitrary units; we treat dist as km for simplicity.
    speeds is array-like length NUM_SENSORS (per-node speeds); for edge i->j use mean speed.
    """
    if not path or len(path) < 2:
        return float('inf')
    total_time = 0.0
    for a, b in zip(path[:-1], path[1:]):
        d = dist_matrix[a, b]
        sp = max(1e-3, (speeds[a] + speeds[b]) / 2.0)
        total_time += d / sp
    return total_time

# Load Data
adj, dist, full_speed_data = load_static_data()
G = None
if build_graph_from_adj is not None:
    try:
        G = build_graph_from_adj(adj, dist)
    except Exception:
        G = None

node_positions = get_node_positions()

# Initialize Model & Device (Early Load)
model, device = load_model()

# Generate Fake "Current" Data for Demo (or load from file)
if 'current_time_step' not in st.session_state:
    st.session_state.current_time_step = full_speed_data.shape[0] - 1

# Trichy Location Mapping (50 Locations) - keep list short if needed
LOCATION_NAMES = [
    "Chatram Bus Stand", "Main Guard Gate", "Thillai Nagar Main Rd", "Fort Station", "Rockfort Temple Base",
    "Gandhi Market", "Palakkarai", "Trichy Railway Station", "Central Bus Stand", "Cantonment",
    "Collector Office Rd", "Court Roundana", "Puthur 4-Roads", "Government Hospital", "Woraiyur",
    "Tennur High Rd", "Srinivasa Nagar", "Vayalur Road", "Bishop Heber College", "K.K. Nagar Arch",
    "TVS Tollgate", "Airport Terminal", "Subramaniyapuram", "Jamal Mohamed College", "Khajamalai",
    "Bharathidasan Univ", "Anna Nagar", "Srirangam Temple", "Amma Mandapam", "Thiruvanaikoil",
    "Cauvery Bridge", "Mambazhasalai", "No 1 Tollgate", "Samayapuram", "NIT Trichy",
    "BHEL Township", "Thuvakudi", "Kattur", "Ariyamangalam", "Ponmalaipatti",
    "Golden Rock", "Senthaneerpuram", "Viralimalai Rd", "Manikandam", "Panjappur",
    "Edamalaipatti Pudur", "Crawford", "Ramalinga Nagar", "Uraiyur Kottai", "Salai Road"
]
if len(LOCATION_NAMES) < NUM_SENSORS:
    LOCATION_NAMES += [f"Location {i}" for i in range(len(LOCATION_NAMES), NUM_SENSORS)]

# UI Layout
st.markdown("<h1 style='text-align: center;'>🌐 UrbanFlow AI Command Center</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #a8b2d1; margin-bottom: 30px;'>Real-Time Traffic Forecasting & Intelligent Routing</h4>", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("🎮 Control Panel")
    st.subheader("Time Travel")
    future_minutes = st.slider("🔮 Prediction Horizon (Min)", 0, 60, 0, step=5)
    st.divider()
    st.subheader("Route Planner")
    loc_map = {name: i for i, name in enumerate(LOCATION_NAMES[:NUM_SENSORS])}
    start_loc = st.selectbox("Start Location", options=LOCATION_NAMES[:NUM_SENSORS], index=0)
    end_loc = st.selectbox("End Destination", options=LOCATION_NAMES[:NUM_SENSORS], index=9)
    start_node = loc_map[start_loc]
    end_node = loc_map[end_loc]
    find_route = st.checkbox("Show Best Route", value=True)
    show_heatmap = st.checkbox("Show Traffic Heatmap", value=True)
    st.divider()
    st.info(f"System Status: {'🟢 GPU' if getattr(device, 'type', '')=='cuda' else '🟡 CPU MODE'}")

# Main Layout
col1, col2 = st.columns([3, 1.5])

# 1. Get Current Traffic State
current_t = st.session_state.current_time_step
if current_t >= full_speed_data.shape[0]:
    current_t = full_speed_data.shape[0] - 1
current_speeds = full_speed_data[current_t].copy()
# historical window
start_idx = max(0, current_t - INPUT_WINDOW + 1)
historical_window = full_speed_data[start_idx:current_t+1]
if historical_window.shape[0] < INPUT_WINDOW:
    # pad with last row
    pad = np.repeat(historical_window[-1:,...], INPUT_WINDOW - historical_window.shape[0], axis=0)
    historical_window = np.vstack([pad, historical_window])

# 2. Predict Future
if torch is not None and hasattr(model, 'to') and hasattr(model, 'eval'):
    # If a real torch model exists, convert and predict. Otherwise use dummy predict.
    try:
        model.eval()
        x_input = (historical_window - historical_window.mean()) / (historical_window.std() + 1e-5)
        if torch is not None:
            x_tensor = torch.FloatTensor(x_input).unsqueeze(0).to('cpu')
            with torch.no_grad():
                pred = model(x_tensor) if callable(model) else model.predict(x_input)
            if isinstance(pred, torch.Tensor):
                predicted_speeds = pred.squeeze().cpu().numpy()
            else:
                predicted_speeds = np.array(pred)
        else:
            predicted_speeds = model.predict(historical_window)
    except Exception:
        predicted_speeds = model.predict(historical_window)
else:
    predicted_speeds = model.predict(historical_window)

display_speeds = predicted_speeds if future_minutes > 0 else current_speeds
display_label = f"Traffic Flow (+{future_minutes} min)" if future_minutes > 0 else "Live Traffic Flow"

# 3. Prepare Node Data for Map
node_data = []
for i in range(NUM_SENSORS):
    lat, lon = node_positions[i]
    node_data.append({
        "lat": lat,
        "lon": lon,
        "name": LOCATION_NAMES[i],
        "speed": float(display_speeds[i]),
        "index": int(i)
    })
df_nodes = pd.DataFrame(node_data)

# Routing: compute standard shortest-distance path and AI path (time-based)
def fallback_route(adj_m, dist_m, start, end, speeds=None):
    Gf = nx.DiGraph()
    N = adj_m.shape[0]
    for i in range(N):
        for j in range(N):
            if adj_m[i, j] > 0:
                # weight by distance or time if speeds provided
                if speeds is None:
                    w = float(dist_m[i, j])
                else:
                    sp = max(1e-3, (speeds[i] + speeds[j]) / 2.0)
                    w = float(dist_m[i, j] / sp)
                Gf.add_edge(i, j, weight=w)
    try:
        path = nx.shortest_path(Gf, source=start, target=end, weight='weight')
        return {"path": path}
    except Exception:
        return {"path": []}

std_path_nodes = []
ai_path_nodes = []
if find_route:
    if calculate_route is not None:
        try:
            res_std = calculate_route(G, start_node, end_node, None)
            std_path_nodes = res_std.get('standard', {}).get('path', [])
            res_ai = calculate_route(G, start_node, end_node, current_speeds, display_speeds)
            ai_path_nodes = res_ai.get('ai', {}).get('path', [])
        except Exception:
            res_std = fallback_route(adj, dist, start_node, end_node, speeds=None)
            res_ai = fallback_route(adj, dist, start_node, end_node, speeds=display_speeds)
            std_path_nodes = res_std['path']
            ai_path_nodes = res_ai['path']
    else:
        res_std = fallback_route(adj, dist, start_node, end_node, speeds=None)
        res_ai = fallback_route(adj, dist, start_node, end_node, speeds=display_speeds)
        std_path_nodes = res_std['path']
        ai_path_nodes = res_ai['path']

# Build PyDeck layers
map_layers = []

# Heatmap / Scatter
if show_heatmap:
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data=df_nodes,
        get_position=["lon", "lat"],
        get_fill_color=["(1 - (speed/80)) * 255", "(speed/80) * 180", 50],
        get_radius=80,
        pickable=True,
        auto_highlight=True
    )
    map_layers.append(scatter)

# Path layers
path_layer_data = []
if std_path_nodes and len(std_path_nodes) >= 2:
    std_coords = [node_positions[n][::-1] for n in std_path_nodes]  # (lon, lat) -> pdk expects [lon,lat]
    std_coords = [[lon, lat] for lat, lon in [node_positions[n] for n in std_path_nodes]]
    path_layer_data.append({"path": std_coords, "color": [255, 50, 50], "width": 40})
if ai_path_nodes and len(ai_path_nodes) >= 2:
    ai_coords = [[lon, lat] for lat, lon in [node_positions[n] for n in ai_path_nodes]]
    path_layer_data.append({"path": ai_coords, "color": [50, 255, 50], "width": 10})

if path_layer_data:
    path_layer = pdk.Layer(
        "PathLayer",
        data=path_layer_data,
        get_path="path",
        get_color="color",
        get_width="width",
        width_scale=1,
        width_min_pixels=5,
        width_max_pixels=30
    )
    map_layers.append(path_layer)

# Recenter Map dynamically based on route
if std_path_nodes:
    route_lats = [node_positions[n][0] for n in std_path_nodes]
    route_lons = [node_positions[n][1] for n in std_path_nodes]
    mid_lat = (max(route_lats) + min(route_lats)) / 2
    mid_lon = (max(route_lons) + min(route_lons)) / 2
else:
    mid_lat, mid_lon = 10.7905, 78.7047

jitter = st.session_state.get("recenter_jitter", 0.0)
view_state = pdk.ViewState(latitude=mid_lat + jitter, longitude=mid_lon + jitter, zoom=12.5, pitch=50)

with col1:
    st.markdown(f"### {display_label}")
    st.pydeck_chart(pdk.Deck(layers=map_layers, initial_view_state=view_state, map_style='light'))
    
    col1_a, col1_b = st.columns([1, 1])
    with col1_a:
        if st.button("🎯 Recenter Map to Route"):
            st.session_state.recenter_jitter = (np.random.rand() - 0.5) * 1e-5
            st.rerun()
    with col1_b:
        start_name = urllib.parse.quote(f"{LOCATION_NAMES[start_node]}, Trichy, Tamil Nadu")
        end_name = urllib.parse.quote(f"{LOCATION_NAMES[end_node]}, Trichy, Tamil Nadu")
        gmaps_link = f"https://www.google.com/maps/dir/?api=1&origin={start_name}&destination={end_name}&travelmode=driving"
        st.markdown(f"**[🗺️ Open Fastest Route in Google Maps (No Stops)]({gmaps_link})**")

# Right column stats & controls
with col2:
    st.metric("Selected Time Index", f"{current_t}")
    avg_speed = float(np.mean(display_speeds))
    st.metric("Average Speed (km/h)", f"{avg_speed:.1f}")
    st.markdown("#### 🚨 Critical Bottlenecks")
    slow_idx = np.argsort(display_speeds)[:5]
    
    # Create intuitive visual progress bars for speed
    for i in slow_idx:
        loc = LOCATION_NAMES[i]
        spd = display_speeds[i]
        st.markdown(f"<span style='color: #e2e8f0; font-weight: 600;'>{loc}</span> <span style='color: #ff4b4b; float: right;'>{spd:.1f} km/h</span>", unsafe_allow_html=True)
        progress = min(max(spd / 80.0, 0.0), 1.0)
        st.progress(progress)
    if st.button("Advance Time"):
        st.session_state.current_time_step = min(full_speed_data.shape[0]-1, st.session_state.current_time_step + 1)
        st.rerun()

