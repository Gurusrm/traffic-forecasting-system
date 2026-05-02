# UrbanFlow AI Command Center

UrbanFlow AI Command Center is a state-of-the-art traffic forecasting and dynamic routing application. Designed for municipal authorities and urban planners, the system leverages Temporal Graph Convolutional Networks (T-GCN) to predict future traffic congestion up to one hour in advance. By combining Graph Convolutional Networks for spatial dependency analysis and Gated Recurrent Units for temporal pattern recognition, UrbanFlow accurately anticipates bottlenecks across city infrastructure before they occur.

The core capability of the UrbanFlow system lies in its AI-enhanced routing engine. Rather than relying solely on distance-based pathfinding, which frequently leads vehicles into congested corridors, the system calculates optimal routes based on predicted travel times. This allows users to visualize and compare standard distance-based routes against intelligent detour strategies that actively bypass predicted congestion hotspots, significantly reducing overall travel time.

The application features a comprehensive interactive dashboard that provides a real-time overview of urban traffic dynamics. Operators can utilize the time-travel simulation to observe how congestion patterns are predicted to shift across the city over the next hour. The interface includes a gradient heatmap to visualize traffic density, a dedicated critical bottlenecks analysis module, and direct integration with Google Maps for seamless navigation handoffs. The current configuration is tailored for the city of Trichy, Tamil Nadu, encompassing fifty distinct real-world landmarks, though the underlying architecture is fully scalable and adaptable to any urban road network.

## Installation and Setup

The system requires Python 3.8 or higher. For accelerated model training and inference, an NVIDIA GPU with the appropriate CUDA-enabled PyTorch installation is recommended but not required.

To deploy the application locally, clone the repository and install the necessary dependencies from the requirements file. 

```bash
git clone https://github.com/yourusername/traffic-prediction-system.git
cd traffic-prediction-system
pip install -r requirements.txt
```

## Usage Guide

Before launching the dashboard, the system must be initialized with traffic pattern data. You can generate synthetic baseline data by executing the generation script.

```bash
python scripts/generate_data.py
```

If you wish to retrain the Temporal Graph Convolutional Network model on custom or updated data, you may run the training module. A pre-trained model checkpoint is already included for immediate use.

```bash
python train.py
```

To access the interactive command center, execute the launcher script appropriate for your operating system.

```bash
# Windows Environments
.\run_dashboard.bat

# Linux and macOS Environments
streamlit run app.py
```

## Architecture

The predictive engine relies on a Temporal Graph Convolutional Network (T-GCN). This architecture processes the topological structure of the road network using Graph Convolution, mapping how traffic flow at intersections influences connected road segments. Simultaneously, it captures temporal dependencies utilizing Gated Recurrent Units to understand how congestion evolves across different times of the day. A fully connected prediction layer ultimately projects these learned spatial-temporal features into accurate future speed predictions for every monitored location in the network.

## License

This software is distributed under the MIT License. Please refer to the LICENSE file for more detailed information regarding distribution and usage rights.
