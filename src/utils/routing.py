import heapq
import numpy as np
import networkx as nx

def build_graph_from_adj(adj_matrix, dist_matrix):
    """
    Constructs a NetworkX graph from adjacency and distance matrices.
    """
    G = nx.Graph()
    num_nodes = adj_matrix.shape[0]
    G.add_nodes_from(range(num_nodes))
    
    rows, cols = np.where(adj_matrix > 0)
    for i, j in zip(rows, cols):
        if i < j: # Undirected, add once
            dist = dist_matrix[i, j]
            G.add_edge(i, j, weight=dist, distance=dist)
            
    return G

def calculate_route(graph, start_node, end_node, current_speeds, predicted_speeds_future=None, avoid_edges=None):
    """
    Calculates the detailed route using Dijkstra.
    
    Args:
        graph: NetworkX graph with 'distance' attribute on edges.
        start_node: int
        end_node: int
        current_speeds: np.array (N,) current speed at each sensor.
        predicted_speeds_future: np.array (N,) predicted speed in future (optional).
        avoid_edges: list of tuples [(u, v), ...] edges to effectively block (optional).
    
    Returns:
        dict: {'path': list, 'duration': float, 'type': str}
    """
    
    def get_travel_time(u, v, d, speeds):
        # Average speed on edge (u, v) is average of speed at u and speed at v
        speed_u = max(speeds[u], 5.0) # Min speed 5 km/h
        speed_v = max(speeds[v], 5.0)
        
        # Check if edge is avoided
        if avoid_edges:
            # Check both directions since graph is undirected but inputs might be ordered
            if (u, v) in avoid_edges or (v, u) in avoid_edges:
                return 100000.0 # Massive penalty, but not infinite (allows path finding)

        avg_speed = (speed_u + speed_v) / 2.0
        
        # Penalize low speeds exponentially to force avoidance of red/orange zones
        # Congestion Thresholds based on Heatmap (80km/h max)
        penalty_factor = 1.0
        if avg_speed < 15.0:
            penalty_factor = 100.0 # Extreme penalty for dead jams (Red)
        elif avg_speed < 35.0:
            penalty_factor = 50.0  # Very Heavy penalty for slow/orange traffic
        elif avg_speed < 55.0:
            penalty_factor = 5.0   # Moderate penalty for medium traffic
        elif avg_speed < 70.0:
            penalty_factor = 1.5   # Slight penalty to prefer max speed
            
        # Scale unit distance to real-world km (approx 20km city scale)
        dist_km = d * 20.0 
        
        # Time (Hours) = Distance (km) / Speed (km/h)
        time_hours = dist_km / avg_speed
        
        # Return Minutes * Penalty
        return (time_hours * 60.0) * penalty_factor

    # 1. Calculate Standard Route (Direct / Shortest Distance)
    # We use distance as weight to simulate the "usual" route people take without traffic info
    standard_weight_fn = lambda u, v, d: d['distance']
    path_standard = run_dijkstra(graph, start_node, end_node, standard_weight_fn)
    
    routes = {
        'standard': {
            'path': path_standard['path'],
            'duration': 0, # Placeholder, will be calculated with real speeds later
            'label': 'Direct Route (Distance)'
        }
    }

    # 2. Calculate AI Route (Future State) if provided
    if predicted_speeds_future is not None:
        ai_weight_fn = lambda u, v, d: get_travel_time(u, v, d['distance'], predicted_speeds_future)
        path_ai = run_dijkstra(graph, start_node, end_node, ai_weight_fn)
        
        routes['ai'] = {
            'path': path_ai['path'],
            'duration': path_ai['cost'],
            'label': 'AI Predicted (Future)'
        }
        
    return routes

def run_dijkstra(graph, start, end, weight_fn):
    """
    Custom Dijkstra to handle dynamic weights.
    weight_fn(u, v, edge_attr) -> cost
    """
    pq = [(0, start, [])] # cost, node, path
    visited = set()
    min_costs = {node: float('inf') for node in graph.nodes}
    min_costs[start] = 0
    
    while pq:
        cost, u, path = heapq.heappop(pq)
        
        if u in visited:
            continue
        visited.add(u)
        
        path = path + [u]
        
        if u == end:
            return {'cost': cost, 'path': path}
        
        for v, attr in graph[u].items():
            if v in visited:
                continue
                
            edge_cost = weight_fn(u, v, attr)
            new_cost = cost + edge_cost
            
            if new_cost < min_costs[v]:
                min_costs[v] = new_cost
                heapq.heappush(pq, (new_cost, v, path))
                
    return {'cost': float('inf'), 'path': []}

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
