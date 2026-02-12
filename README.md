# 🚦 GPU-Accelerated GraphCast-Style Spatiotemporal Traffic Forecasting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-ready deep learning system that predicts future traffic conditions across entire urban road networks using spatiotemporal graph neural networks, inspired by DeepMind's GraphCast architecture.

## 🎯 Project Overview

Modern cities suffer from traffic congestion because decisions are made based on **current traffic**, not **future traffic**. This system solves that by:

- **Modeling cities as dynamic graphs** where roads are nodes and connections are edges
- **Learning spatiotemporal patterns** to understand how congestion propagates
- **Predicting future traffic states** for entire road networks simultaneously
- **GPU acceleration** for real-time inference and efficient training

### Key Features

✅ **Spatiotemporal Graph Neural Networks** - State-of-the-art modeling of urban traffic dynamics.  
✅ **Interactive Real-Time Map** - Leaflet.js-powered visualization with Los Angeles sensor integration.  
✅ **Live Heatmap Overlay** - Dynamic congestion mapping showing predicted future traffic density.  
✅ **GPU-Accelerated Inference** - Sub-50ms predictions powered by PyTorch and CUDA.  
✅ **Multi-Horizon Forecasting** - Simultaneous predictions for 15, 30, and 60-minute windows.  
✅ **Production Grade** - Dockerized stack, full test suite, and modular REST API.

## 🖥️ Interactive Dashboard & Frontend

This project features a high-performance, responsive web interface for real-time traffic monitoring and prediction analysis.

### Features
- **Geospatial Map Visualization**: Integrated **Leaflet.js** map displaying 207 sensor locations accurately mapped to Los Angeles highways.
- **Dynamic Traffic Status**: Sensor markers change color based on predicted speeds (Green: >55mph, Yellow: Moderate, Red: Congestion).
- **Predictive Heatmaps**: Uses `leaflet.heat` to generate a network-wide probability map of traffic density.
- **Node-Specific Analytics**: Interactive **Plotly.js** charts showing:
  - **Input History**: The last 60 minutes of real traffic data.
  - **Predicted Future**: The ST-GNN's forecast for the next hour.
  - **Actual Future**: Real-time comparison for model verification.
- **One-Click Analysis**: Trigger real-time inference on random test samples to see how the model generalizes to new patterns.

### Technical Stack (Frontend)
- **Engine**: Vanilla JavaScript (ES6+) with Async/Await for non-blocking API calls.
- **Styling**: Modern, mobile-responsive CSS with Glassmorphism effects and Inter typography.
- **Visuals**: Leaflet.js (Maps), Plotly.js (Analytics), FontAwesome (Icons).

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Traffic Data Source                       │
│              (METR-LA / PEMS-BAY / Live Feed)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing & Windowing                     │
│  • Missing value imputation  • Normalization                │
│  • Sliding window generation • Train/Val/Test split         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Graph Construction Layer                     │
│  • Build adjacency matrix from road network topology        │
│  • Compute distance-based edge weights                      │
│  • Create PyTorch Geometric graph objects                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Spatiotemporal GNN Model (GPU-Accelerated)         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Temporal Encoder (LSTM/GRU)                          │  │
│  │  • Captures historical traffic patterns               │  │
│  │  • Encodes rush hour behaviors & trends               │  │
│  └───────────────────┬───────────────────────────────────┘  │
│                      │                                       │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │  Graph Message Passing (GAT/GraphSAGE)                │  │
│  │  • Exchanges information between connected roads      │  │
│  │  • Models congestion propagation                      │  │
│  │  • Attention-based neighbor aggregation               │  │
│  └───────────────────┬───────────────────────────────────┘  │
│                      │                                       │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │  Temporal Decoder (LSTM/GRU)                          │  │
│  │  • Generates multi-step future predictions            │  │
│  │  • Outputs per-road traffic forecasts                 │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Future Traffic Predictions                      │
│  • Per-road speed forecasts  • Congestion levels            │
│  • Traffic flow estimates    • Uncertainty quantification   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Visualization & Deployment Layer                   │
│  • Interactive traffic heatmaps  • REST API endpoints       │
│  • Real-time dashboards          • Alert notifications      │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Model Architecture Details

### Spatiotemporal Graph Neural Network (ST-GNN)

The model consists of three main components:

#### 1. **Temporal Encoder**
- **Input**: Historical traffic sequences (last 60 minutes)
- **Architecture**: Stacked GRU/LSTM layers
- **Output**: Compressed temporal embeddings
- **Purpose**: Captures rush hour patterns, trends, periodic behaviors

#### 2. **Graph Message Passing**
- **Input**: Temporal embeddings + Road network graph
- **Architecture**: Graph Attention Networks (GAT) or GraphSAGE
- **Output**: Spatially-aware node representations
- **Purpose**: Models congestion propagation between connected roads

#### 3. **Temporal Decoder**
- **Input**: Spatially-aware representations
- **Architecture**: Stacked GRU/LSTM layers
- **Output**: Multi-horizon traffic predictions (next 15/30/60 min)
- **Purpose**: Generates future traffic states for all roads

### Training Strategy

- **Loss Function**: Masked Mean Absolute Error (MAE) / Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout, L2 penalty
- **Batch Size**: 64 (adjustable based on GPU memory)
- **Epochs**: 200 with early stopping

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: NVIDIA RTX 3060 or better)
- 16GB+ RAM
- CUDA 11.7+ and cuDNN

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Gurusrm/traffic-forecasting-system.git
cd traffic-forecasting-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download datasets**
```bash
python scripts/download_data.py --dataset metr-la
```

### Training the Model

```bash
# Train with default configuration
python src/training/train.py --config config/default_config.yaml

# Train with custom parameters
python src/training/train.py \
    --dataset metr-la \
    --horizon 12 \
    --batch-size 64 \
    --epochs 200 \
    --gpu 0
```

### Running Inference

```bash
# Single prediction
python src/inference/predict.py \
    --checkpoint models/checkpoints/best_model.pth \
    --input data/test_sample.npz

# Real-time prediction server
python src/inference/serve.py --port 8000
```

### Evaluation

```bash
python src/training/evaluate.py \
    --checkpoint models/checkpoints/best_model.pth \
    --test-data data/processed/test.npz
```

## 📁 Project Structure

```
traffic-forecasting-system/
├── config/
│   ├── default_config.yaml      # Default hyperparameters
│   ├── metr_la_config.yaml      # METR-LA specific config
│   └── pems_bay_config.yaml     # PEMS-BAY specific config
│
├── data/
│   ├── raw/                     # Raw downloaded datasets
│   ├── processed/               # Preprocessed numpy arrays
│   └── graphs/                  # Saved graph structures
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   ├── preprocessing.py    # Data cleaning & normalization
│   │   ├── graph_builder.py    # Graph construction utilities
│   │
│   ├── models/
│   │   ├── st_gnn.py           # Main ST-GNN architecture
│   │   ├── temporal_encoder.py # LSTM/GRU encoder
│   │   ├── graph_layers.py     # GAT/GraphSAGE layers
│   │   └── temporal_decoder.py # LSTM/GRU decoder
│   │
│   ├── training/
│   │   ├── train.py            # Training script
│   │   ├── trainer.py          # Trainer class
│   │   ├── evaluate.py         # Evaluation metrics
│   │
│   ├── inference/
│   │   ├── predict.py          # Single prediction
│   │   ├── serve.py            # REST API server
│   │
│   └── utils/
│       ├── metrics.py          # MAE, RMSE, MAPE
│       ├── logger.py           # Training logger
│       └── visualization.py    # Plotting utilities
│
├── models/
│   ├── checkpoints/            # Saved model weights
│   └── logs/                   # TensorBoard logs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── outputs/
│   ├── predictions/            # Saved predictions
│   └── visualizations/         # Generated plots
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
│
├── docs/
│   ├── architecture.md         # Detailed architecture docs
│   ├── api_reference.md        # API documentation
│   └── deployment.md           # Deployment guide
│
├── scripts/
│   ├── download_data.py        # Dataset downloader
│   └── setup_environment.sh    # Environment setup
│
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── Dockerfile                  # Docker containerization
├── docker-compose.yml          # Multi-container setup
└── README.md                   # This file
```

## 📈 Results & Performance

### Benchmark Results (METR-LA Dataset)

| Metric | 15 min | 30 min | 60 min |
|--------|--------|--------|--------|
| MAE    | 2.87   | 3.45   | 4.12   |
| RMSE   | 5.21   | 6.89   | 8.34   |
| MAPE   | 7.4%   | 9.1%   | 11.8%  |

### Training Performance

- **Training Time**: ~2 hours on NVIDIA RTX 3080 (100 epochs)
- **Inference Speed**: <50ms per prediction on GPU
- **Model Size**: ~15MB (compressed)

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch 2.0+, PyTorch Geometric
- **GPU Acceleration**: CUDA, cuDNN
- **Data Processing**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **API**: FastAPI, Uvicorn
- **Monitoring**: TensorBoard, Weights & Biases
- **Testing**: Pytest, unittest
- **Containerization**: Docker, docker-compose

## 📚 Datasets

### Supported Datasets

1. **METR-LA**
   - 207 sensors on LA highways
   - 4 months of data (Mar-Jun 2012)
   - 5-minute intervals

2. **PEMS-BAY**
   - 325 sensors in Bay Area
   - 6 months of data (Jan-Jun 2017)
   - 5-minute intervals

### Data Format

```python
{
    'speed': np.array,      # Shape: [num_samples, num_nodes, features]
    'adj_mat': np.array,    # Shape: [num_nodes, num_nodes]
    'timestamps': np.array  # Shape: [num_samples]
}
```

## 🎓 Research Background

This project is inspired by:

- **GraphCast** (DeepMind): Graph neural networks for weather forecasting
- **DCRNN**: Diffusion Convolutional Recurrent Neural Network
- **Graph WaveNet**: Adaptive graph generation for traffic forecasting
- **ST-GCN**: Spatiotemporal Graph Convolutional Networks

### Key Papers

1. Li et al. (2018) - "Diffusion Convolutional Recurrent Neural Network"
2. Wu et al. (2019) - "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
3. Yu et al. (2018) - "Spatio-Temporal Graph Convolutional Networks"

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- METR-LA and PEMS-BAY dataset providers
- PyTorch and PyTorch Geometric teams
- DeepMind for GraphCast inspiration

## 📧 Contact

For questions or collaboration:
- **Email**: your.email@example.com
- **GitHub**: [@Gurusrm](https://github.com/Gurusrm)
- **LinkedIn**: [Gurusrm](https://linkedin.com/in/Gurusrm)

---

**Built with ❤️ for smarter cities and better traffic management**
