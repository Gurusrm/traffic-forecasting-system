
import sys
import os
import numpy as np
import networkx as nx
import pandas as pd

# Add root to path
sys.path.append(os.path.abspath('.'))

from src.utils.routing import build_graph_from_adj, calculate_route

DATA_DIR = 'data/processed'
NUM_SENSORS = 50

def get_node_positions():
    np.random.seed(42)
    node_pos = np.random.rand(NUM_SENSORS, 2)
    lat_base, lon_base = 10.7905, 78.7047
    lat_spread, lon_spread = 0.1, 0.1
    positions = {}
    for i in range(NUM_SENSORS):
        lat = lat_base + (node_pos[i, 0] - 0.5) * lat_spread
        lon = lon_base + (node_pos[i, 1] - 0.5) * lon_spread
        positions[i] = [lon, lat]
    return positions

def main():
    print("Loading data...")
    adj = np.load(os.path.join(DATA_DIR, 'adj_matrix.npy'))
    dist = np.load(os.path.join(DATA_DIR, 'dist_matrix.npy'))
    
    print("Building graph...")
    G = build_graph_from_adj(adj, dist)
    node_positions = get_node_positions()
    
    start_node = 0
    end_node = 9
    
    print(f"Calculating route from {start_node} to {end_node}...")
    standard_route_res = calculate_route(G, start_node, end_node, None)
    std_path_nodes = standard_route_res['standard']['path']
    print(f"Standard path: {std_path_nodes}")
    
    routing_speeds = np.ones(NUM_SENSORS) * 80.0
    
    print("Simulating Blast Radius...")
    try:
        # Blast Logic from app.py
        mid_idx = len(std_path_nodes) // 2
        center_node = std_path_nodes[mid_idx]
        center_pos = node_positions[center_node]
        
        blast_radius = 0.05
        print(f"Center Node: {center_node}, Pos: {center_pos}, Radius: {blast_radius}")
        
        jammed_count = 0
        for i in range(NUM_SENSORS):
            pos = node_positions[i]
            dist = np.sqrt((pos[0] - center_pos[0])**2 + (pos[1] - center_pos[1])**2)
            if dist < blast_radius:
                if i != start_node and i != end_node:
                    routing_speeds[i] = 1.0
                    jammed_count += 1
        print(f"Jammed {jammed_count} nodes.")
        
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
