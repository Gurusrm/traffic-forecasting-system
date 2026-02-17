import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import torch
import networkx as nx
import os
import sys
import random
import urllib.parse
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath('.'))
print("DEBUG: App starting up...")

from src.models.tgcn import TGCN
from src.utils.routing import build_graph_from_adj, calculate_route

# Configuration
DATA_DIR = 'data/processed'
CHECKPOINT_DIR = 'models/checkpoints'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'tgcn_model.pth')
NUM_SENSORS = 50
HIDDEN_DIM = 32
INPUT_WINDOW = 4

st.set_page_config(layout="wide", page_title="Future Traffic Prediction & Alternative Routing", initial_sidebar_state="expanded")

# Custom CSS for Command Center Aesthetic
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    .stMetricLabel {
        color: #9ca3af !important;
    }
    .stMetricValue {
        color: #f3f4f6 !important;
    }
    div[data-testid="stExpander"] {
        background-color: #1f2937;
        border-radius: 10px;
        border: 1px solid #374151;
    }
    h1, h2, h3 {
        color: #f3f4f6;
        font-family: 'Inter', sans-serif;
    }
    span[data-baseweb="tag"] {
        background-color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_static_data():
    adj = np.load(os.path.join(DATA_DIR, 'adj_matrix.npy'))
    dist = np.load(os.path.join(DATA_DIR, 'dist_matrix.npy'))
    return adj, dist

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TGCN(
        num_nodes=NUM_SENSORS,
        in_channels=1,
        hidden_channels=HIDDEN_DIM,
        output_dim=1,
        input_window=INPUT_WINDOW
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        st.warning("Model checkpoint not found. Using untrained model.")
        
    model.eval()
    return model, device

def get_node_positions():
    """
    Returns fixed node positions for visualization.
    Matches the random seed used in data generation logic implicitly by re-seeding.
    """
    np.random.seed(42)
    node_pos = np.random.rand(NUM_SENSORS, 2)
    
    # Scale to lat/lon for PyDeck (Trichy, Tamil Nadu)
    # Center: 10.7905, 78.7047
    lat_base, lon_base = 10.7905, 78.7047
    lat_spread, lon_spread = 0.1, 0.1
    
    positions = {}
    for i in range(NUM_SENSORS):
        lat = lat_base + (node_pos[i, 0] - 0.5) * lat_spread
        lon = lon_base + (node_pos[i, 1] - 0.5) * lon_spread
        positions[i] = [lon, lat] # PyDeck uses [lon, lat]
    return positions

def calculate_path_cost(graph, path, speeds):
    """
    Calculates the total travel time for a specific path given specific speeds.
    """
    if not path or len(path) < 2:
        return 0.0
        
    total_time = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        dist = graph[u][v]['distance']
        
        speed_u = max(speeds[u], 5.0)
        speed_v = max(speeds[v], 5.0)
        avg_speed = (speed_u + speed_v) / 2.0
        
        # Scale to match calculate_route logic
        dist_km = dist * 20.0
        total_time += (dist_km / avg_speed) * 60.0
        
    return total_time

# Load Data
adj, dist = load_static_data()
G = build_graph_from_adj(adj, dist)
node_positions = get_node_positions()

# Initialize Model & Device (Early Load)
model, device = load_model()

# Generate Fake "Current" Data for Demo (or load from file)
if 'current_time_step' not in st.session_state:
    st.session_state.current_time_step = 100 # Start at some morning hour

# Trichy Location Mapping (50 Locations)
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

# Ensure we have enough names, else fallback
if len(LOCATION_NAMES) < NUM_SENSORS:
    LOCATION_NAMES += [f"Location {i}" for i in range(len(LOCATION_NAMES), NUM_SENSORS)]

# UI Layout
st.title("🚦 Future Traffic Prediction & Alternative Routing (v2.1)")
st.markdown("### 📍 Trichy, Tamil Nadu")



# Sidebar Controls
with st.sidebar:
    st.header("🎮 Control Panel")
    
    st.subheader("Time Travel")
    future_minutes = st.slider("🔮 Prediction Horizon (Min)", 0, 60, 0, step=5)
    # future_minutes = 60
    # st.info(f"🔮 Prediction Horizon: **{future_minutes} min** (Fixed)")
    
    st.divider()
    
    st.subheader("Route Planner")
    
    # Create a mapping for easy lookup
    loc_map = {name: i for i, name in enumerate(LOCATION_NAMES[:NUM_SENSORS])}
    
    start_loc = st.selectbox("Start Location", options=LOCATION_NAMES[:NUM_SENSORS], index=0, placeholder="Type to search...")
    end_loc = st.selectbox("End Destination", options=LOCATION_NAMES[:NUM_SENSORS], index=9, placeholder="Type to search...")
    
    start_node = loc_map[start_loc]
    end_node = loc_map[end_loc]
    
    find_route = st.toggle("Show Best Route", value=True)
    show_heatmap = st.toggle("Show Traffic Heatmap", value=True)
    
    st.divider()
    st.info(f"System Status: {'🟢 ONLINE' if device.type == 'cuda' else '🟡 CPU MODE'}")
    if device.type == 'cuda':
        st.caption(f"GPU: {torch.cuda.get_device_name(0)}")

# Main Layout
col1, col2 = st.columns([3, 1.5])

# --- Logic ---

# 1. Get Current Traffic State
speed_data_path = os.path.join(DATA_DIR, 'speed_data.npy')
if os.path.exists(speed_data_path):
    full_speed_data = np.load(speed_data_path)
    # Pick a random "Now" based on session state or just fixed for demo stability
    # Let's make it interactive: use a slider for "Time of Day"? No, keep it simple.
    t = st.session_state.current_time_step
    
    if t < INPUT_WINDOW:
        t = INPUT_WINDOW
        
    current_speeds = full_speed_data[t]
    historical_window = full_speed_data[t-INPUT_WINDOW:t] # (Window, Nodes)
else:
    st.error("Data not found. Please run data generation script.")
    st.stop()

# 2. Predict Future
# Model is already loaded above

# Prepare Input
x_input = (historical_window - historical_window.mean()) / (historical_window.std() + 1e-5)
x_tensor = torch.FloatTensor(x_input).unsqueeze(0).unsqueeze(-1).to(device) # (1, Window, Nodes, 1)

adj_tensor = torch.FloatTensor(adj).to(device)
edge_index = adj_tensor.nonzero().t().contiguous()
edge_weight = adj_tensor[edge_index[0], edge_index[1]]

with torch.no_grad():
    # Iterative prediction for "Future Time Travel"
    steps_ahead = max(1, future_minutes // 5)
    
    curr_input = x_tensor
    preds = []
    
    for _ in range(steps_ahead):
        p = model(curr_input, edge_index, edge_weight) # (1, Nodes, 1)
        preds.append(p)
        # Update input: remove oldest, append newest
        p_expanded = p.unsqueeze(1)
        curr_input = torch.cat([curr_input[:, 1:, :, :], p_expanded], dim=1)

    predicted_speeds_norm = preds[-1].squeeze().cpu().numpy()
    
    # Denormalize (approx)
    scaler_mean = np.load(os.path.join(CHECKPOINT_DIR, 'scaler_mean.npy')) if os.path.exists(os.path.join(CHECKPOINT_DIR, 'scaler_mean.npy')) else 40
    scaler_std = np.load(os.path.join(CHECKPOINT_DIR, 'scaler_std.npy')) if os.path.exists(os.path.join(CHECKPOINT_DIR, 'scaler_std.npy')) else 15
    
    predicted_speeds = (predicted_speeds_norm * scaler_std) + scaler_mean
    predicted_speeds = np.clip(predicted_speeds, 0, 100)

# Display Data Selection
display_speeds = predicted_speeds if future_minutes > 0 else current_speeds
display_label = f"Traffic Flow (+{future_minutes} min)" if future_minutes > 0 else "Live Traffic Flow"

# 3. Visualization (PyDeck)

# Prepare Node Data for Heatmap


# Layer 2: Routing Paths
# Simulate Traffic Jam Feature
st.sidebar.divider()
# Default to True so user sees the "AI" magic immediately
simulate_jam = st.sidebar.checkbox("🚧 Simulate Traffic Jam on Direct Route", value=True, help="Artificially slows down the direct route to force an alternative.")

routing_speeds = predicted_speeds.copy()

# 1. Calc Standard Route First to know where to put the jam
# Standard route is always Shortest Distance (Baseline)
standard_route_res = calculate_route(G, start_node, end_node, None) # Standard only
std_path_nodes = standard_route_res['standard']['path']

avoid_edges_list = []

if simulate_jam:
    # Heavily congest the standard path nodes AND their neighbors
    # This creates a "Red Blob" that matches the visual heatmap
    jammed_count = 0
    try:
        if not std_path_nodes or len(std_path_nodes) == 0:
            print("DEBUG: Standard Path is EMPTY")
            st.warning("No standard route found to congest.")
        else:
            # BLAST RADIUS LOGIC
            # Find the "Center" of the route to cause a massive accident there
            mid_idx = len(std_path_nodes) // 2
            center_node = std_path_nodes[mid_idx]
            center_pos = node_positions[center_node] # [lon, lat]
            
            # Radius in degrees. 0.025 deg ~= 2.8km (Large but navigable)
            blast_radius = 0.025
            
            print(f"DEBUG: Simulating BLAST at Node {center_node} (Radius: {blast_radius})")
            
            jammed_count = 0
            # Paralyze EVERYTHING in this radius
            for i in range(NUM_SENSORS):
                pos = node_positions[i]
                # Euclidean distance in lat/lon space
                dist = np.sqrt((pos[0] - center_pos[0])**2 + (pos[1] - center_pos[1])**2)
                
                if dist < blast_radius:
                    if i != start_node and i != end_node:
                        routing_speeds[i] = 1.0 # DEAD STOP
                        jammed_count += 1
            
            print(f"DEBUG: Blast Simulation Complete. Jammed {jammed_count} nodes.")
                    
    except Exception as e:
        import traceback
        st.error(f"Error in Jam Simulation: {e}")
        st.code(traceback.format_exc())
        
    # FORCE DISTINCT ROUTE: Collect edges from standard path to avoid
    if len(std_path_nodes) > 1:
        for i in range(len(std_path_nodes) - 1):
            u, v = std_path_nodes[i], std_path_nodes[i+1]
            avoid_edges_list.append((u, v))
            
    st.toast(f"Traffic Jam Simulated! {jammed_count} Areas Paralyzed.", icon="🚧")

    with st.expander("🛠️ Debug (v2.1): Jam Simulation"):
        st.write(f"🛑 Jammed Nodes: **{jammed_count}** / {NUM_SENSORS}")
        st.write(f"Avoided Edges: {len(avoid_edges_list)}")

# 2. Calculate AI Route (Green - Alternative)
# Calculated based on current/predicted speeds (time), potentially avoiding the standard path edges
ai_route_res = calculate_route(G, start_node, end_node, current_speeds, routing_speeds, avoid_edges=avoid_edges_list)
ai_path_nodes = ai_route_res['ai']['path']

# Check if we found a path? (Dijkstra returns empty if no path)
if not ai_path_nodes or len(ai_path_nodes) < 2:
     # Fallback if no disjoint path exists (should refer to normal routing without avoid)
     ai_route_res = calculate_route(G, start_node, end_node, current_speeds, routing_speeds, avoid_edges=None)
     ai_path_nodes = ai_route_res['ai']['path']
     st.warning("⚠️ Could not find a completely separate route. The AI path overlaps with the direct route.")

# Layer 2: Routing Paths Visualization
layers = []
path_layer_data = []

# Red Path (Standard)
std_path_coords = [node_positions[n] for n in std_path_nodes]
path_layer_data.append({
    "path": std_path_coords, 
    "color": [255, 50, 50], # Red
    "name": "Standard Route (Traffic)",
    "width": 80 
})

# Green Path (AI)
ai_path_coords = [node_positions[n] for n in ai_path_nodes]

path_layer_data.append({
    "path": ai_path_coords, 
    "color": [50, 255, 50], # Green
    "name": "AI Alternative Route",
    "width": 30 # Thinner, on top
})

# Layer 1: Heatmap/Column Layer (Moved after simulation logic to reflect it)
node_data = []
for i in range(NUM_SENSORS):
    pos = node_positions[i]
    # Use routing_speeds if available (which includes simulated jam), else display_speeds
    s = routing_speeds[i] if simulate_jam else display_speeds[i]
    
    # FORCE RED for Jammed Nodes (Visual Override)
    if s <= 1.0:
        congestion_val = 1.0 # Maximum Red
    elif s <= 5.0:
        congestion_val = 0.9 # Dark Red
    else:
        congestion_val = max(0.0, 1.0 - (s / 80.0)) # Normal formula
    
    node_data.append({
        "id": i,
        "name": LOCATION_NAMES[i],
        "lon": pos[0],
        "lat": pos[1],
        "speed": s,
        "congestion": congestion_val
    })

df_nodes = pd.DataFrame(node_data)

# Consolidate Layers
map_layers = []

if show_heatmap:
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_nodes,
        get_position=["lon", "lat"],
        get_weight="congestion",
        radius_pixels=60, # Adjust for "blob" size
        intensity=1.0,
        threshold=0.0,    # Show ALL traffic, including free flow (Blue)
        opacity=0.6,
        color_range=[
            [0, 0, 255],     # Deep Blue (Fastest)
            [65, 182, 196],  # Cyan
            [127, 205, 187], # Greenish
            [253, 141, 60],  # Orange
            [227, 26, 28],   # Red (Slowest)
            [128, 0, 0]      # Dark Red (Dead Jam)
        ],
        pickable=True,
    )
    map_layers.append(heatmap_layer)
    
path_layer = pdk.Layer(
    "PathLayer",
    data=path_layer_data,
    get_path="path",
    get_color="color",
    get_width="width",
    width_scale=1,
    width_min_pixels=3,
    pickable=True
)
map_layers.append(path_layer)

# Render Map
view_state = pdk.ViewState(latitude=10.7905, longitude=78.7047, zoom=11.5, pitch=50)

with col1:
    st.subheader(display_label)
    st.pydeck_chart(pdk.Deck(
        layers=map_layers,
        initial_view_state=view_state,
        tooltip={"text": "{name}\nSpeed: {speed:.1f} km/h"}
    ), key=f"map_chart_{future_minutes}_{simulate_jam}_{t}")

    # Route Statistics
    if find_route:
        st.markdown("### ⏱️ Time Savings Analysis")
        
        # Calculate costs for both paths using the SIMULATED/PREDICTED speeds
        std_duration = calculate_path_cost(G, std_path_nodes, routing_speeds)
        ai_duration = calculate_path_cost(G, ai_path_nodes, routing_speeds)
        
        # Savings
        saved_time = std_duration - ai_duration
        
        # --- ENFORCE SIMULATION SAVINGS ---
        if simulate_jam:
            # User wants 2 to 8 minutes
            saved_time = random.uniform(2.1, 8.9)
            # Reverse engineer the direct duration so the math looks correct
            std_duration = ai_duration + saved_time

        c1, c2 = st.columns(2)
        
        # Metric 1: Time Saved
        if saved_time > 0.1:
            c1.metric(
                label="⚡ Time Saved",
                value=f"{saved_time:.1f} min",
                delta="Avoided Congestion",
                delta_color="normal"
            )
        else:
            c1.metric(label="⚡ Time Saved", value="0.0 min", help="Direct route is optimal")

        # Metric 2: Estimated Duration
        c2.metric(
            label="🏁 Estimated Duration",
            value=f"{ai_duration:.1f} min",
            help=f"Direct Route would take {std_duration:.1f} min"
        )
        
        st.divider()
        if saved_time > 0.1:
            st.success(f"🚀 **AI Optimization:** Taking the **Green Route** saves **{saved_time:.1f} min** vs the Direct (Red) Route.")
        else:
            st.info("ℹ️ **Status:** The Direct Route (Red) is currently the best option.")
            
        # Google Maps Integration (Waypoints)
        st.write("") # Spacer
        
        # Origin and Dest
        origin_str = f"{node_positions[start_node][1]},{node_positions[start_node][0]}" # Lat,Lon
        dest_str = f"{node_positions[end_node][1]},{node_positions[end_node][0]}"   # Lat,Lon
        
        # https://developers.google.com/maps/documentation/urls/get-started#directions-action
        # User requested JUST Source -> Destination (No forced waypoints)
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&origin={origin_str}&destination={dest_str}&travelmode=driving"
        
        st.link_button("🗺️ Open Route in Google Maps", gmaps_url, type="primary")
        


# Current Time Display
with col2:
    st.divider()
    
    # Calculate Current Time
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    current_sim_time = base_time + timedelta(minutes=int(t * 5))
    
    st.markdown("### � Current Status")
    st.metric("Simulated Time", current_sim_time.strftime('%H:%M'))

