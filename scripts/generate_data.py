import numpy as np
import pandas as pd
import os
import networkx as nx

def generate_traffic_data(
    num_sensors=50,
    num_days=7,
    interval_min=5,
    output_dir='data/processed'
):
    """
    Generates synthetic traffic data:
    1. Graph structure (Adjacency Matrix)
    2. Speed data (Time, Sensors) with patterns and anomalies
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Time settings
    intervals_per_day = (24 * 60) // interval_min
    total_steps = num_days * intervals_per_day
    
    print(f"Generating data for {total_steps} time steps ({num_days} days)...")

    # 1. Generate Graph (Geometric Graph for spatial locality)
    # Positions in a normalized 2D space
    np.random.seed(42) # Ensure graph topology matches visualization in app.py
    pos = np.random.rand(num_sensors, 2)
    # Create graph based on distance threshold
    threshold = 0.8 # Ultra-high density to guarantee redundancy
    adj_matrix = np.zeros((num_sensors, num_sensors))
    
    G = nx.Graph()
    G.add_nodes_from(range(num_sensors))
    
    dist_matrix = np.zeros((num_sensors, num_sensors))

    for i in range(num_sensors):
        for j in range(num_sensors):
            if i != j:
                dist = np.linalg.norm(pos[i] - pos[j])
                dist_matrix[i, j] = dist
                if dist < threshold:
                    adj_matrix[i, j] = 1
                    G.add_edge(i, j, weight=dist)

    # Ensure connected graph (naive fix: connect disconnected components)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for k in range(len(components) - 1):
            u = list(components[k])[0]
            v = list(components[k+1])[0]
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
            G.add_edge(u, v, weight=np.linalg.norm(pos[u] - pos[v]))

    # Save Adjacency Matrix
    np.save(os.path.join(output_dir, 'adj_matrix.npy'), adj_matrix)
    np.save(os.path.join(output_dir, 'dist_matrix.npy'), dist_matrix)
    print(f"Graph generated with {G.number_of_edges()} edges.")

    # 2. Generate Speed Data
    # Base Speed: ~60 km/h (free flow)
    # Patterns: 
    #   - Morning Rush (7-9 AM): Drops to ~20-30 km/h
    #   - Evening Rush (5-7 PM): Drops to ~20-30 km/h
    #   - Night (11 PM - 5 AM): High speed ~65 km/h
    #   - Weekend: Less congestion
    
    speed_data = np.zeros((total_steps, num_sensors))
    
    for t in range(total_steps):
        # Determine time of day and day of week
        current_minute = t * interval_min
        day_idx = current_minute // (24 * 60)
        minute_of_day = current_minute % (24 * 60)
        hour_of_day = minute_of_day / 60.0
        
        is_weekend = day_idx >= 5 # 0=Mon, ..., 4=Fri, 5=Sat, 6=Sun
        
        # Base Noise
        base_speed = 60 + np.random.normal(0, 2, num_sensors)
        
        congestion_factor = np.zeros(num_sensors)
        
        if not is_weekend:
            # Morning Rush (7-9 AM)
            if 7 <= hour_of_day <= 9:
                # Gaussian peak at 8 AM
                intensity = np.exp(-0.5 * ((hour_of_day - 8) / 1.0)**2)
                congestion_factor += intensity * 35 # Drop by up to 35 km/h
            
            # Evening Rush (5-7 PM)
            elif 17 <= hour_of_day <= 19:
                 # Gaussian peak at 6 PM
                intensity = np.exp(-0.5 * ((hour_of_day - 18) / 1.0)**2)
                congestion_factor += intensity * 40 # Drop by up to 40 km/h
        else:
            # Weekend mild congestion
            if 11 <= hour_of_day <= 18:
                 intensity = np.exp(-0.5 * ((hour_of_day - 14) / 3.0)**2)
                 congestion_factor += intensity * 10

        # Apply spatial smoothing to congestion (traffic jams spread)
        # Simple diffusion step
        congestion_factor = 0.7 * congestion_factor + 0.3 * (adj_matrix @ congestion_factor) / (adj_matrix.sum(axis=1) + 1e-5)
        
        current_speeds = base_speed - congestion_factor
        current_speeds = np.clip(current_speeds, 5, 80) # Clip between 5 and 80 km/h
        
        speed_data[t] = current_speeds

    # 3. Anomalies (Accidents)
    # Randomly pick time and location
    num_anomalies = 20 # Increased from 10
    for _ in range(num_anomalies):
        t_start = np.random.randint(0, total_steps - 24) # At least 2 hour duration
        sensor_idx = np.random.randint(0, num_sensors)
        duration = np.random.randint(12, 36) # 1 to 3 hours
        
        # Traffic blockage spreads to neighbors
        neighbors = np.where(adj_matrix[sensor_idx] > 0)[0]
        affected_sensors = [sensor_idx] + list(neighbors)
        
        for t in range(t_start, t_start + duration):
            for s in affected_sensors:
                # Severe congestion
                drop = 55 if s == sensor_idx else 40
                speed_data[t, s] = max(5, speed_data[t, s] - drop)

    # Save Speed Data
    np.save(os.path.join(output_dir, 'speed_data.npy'), speed_data)
    print("Speed data generated.")
    
    # Save a CSV for easy inspection (first 5 sensors)
    df = pd.DataFrame(speed_data[:, :5], columns=[f'Sensor_{i}' for i in range(5)])
    df['Time_Step'] = np.arange(total_steps)
    df.to_csv(os.path.join(output_dir, 'speed_data_sample.csv'), index=False)
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    generate_traffic_data()
