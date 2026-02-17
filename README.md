# 🚦 Future Traffic Prediction & Alternative Routing System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A state-of-the-art traffic forecasting and dynamic routing application powered by **Temporal Graph Convolutional Networks (T-GCN)**. This system predicts future traffic congestion up to 1 hour in advance and suggests AI-optimized alternative routes to save travel time.

---

## 🌟 Key Features

### 1. 🔮 Future Traffic Prediction
- **T-GCN Model:** Combines Graph Convolutional Networks (GCN) for spatial dependencies and Gated Recurrent Units (GRU) for temporal patterns.
- **Horizon:** Predicts traffic speeds and congestion levels **60 minutes into the future**.
- **Accuracy:** Trained on historical traffic patterns including rush hours, weekends, and anomalies.

### 2. 🛣️ AI-Enhanced Routing
- **Direct vs. AI Route:** 
    - **🔴 Direct Route (Red):** The standard shortest path based on distance, often congested.
    - **🟢 AI Route (Green):** The intelligent path that avoids predicted congestion.
- **Time Savings:** Real-time calculation of time saved by choosing the AI route.

### 3. 🗺️ Interactive Command Center
- **2D Gradient Heatmap:** Visualizes traffic density across the city (Blue = Fast, Red = Congested).
- **Time Travel:** Simulate future traffic conditions with a sliding timeline.
- **Google Maps Integration:** One-click button to open the selected route directly in Google Maps for navigation.

### 4. 📍 Real-World Context
- **Location Config:** Pre-configured for **Trichy, Tamil Nadu**, with 50 real-world landmarks (e.g., Chatram Bus Stand, Thillai Nagar, Rockfort Temple).
- **Scalable:** Can be adapted to any city by updating the graph structure.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (Optional, for faster training)

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/traffic-prediction-system.git
   cd traffic-prediction-system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU support, ensure you have the correct PyTorch version with CUDA installed.*

---

## 🚀 Usage Guide

### 1. Generate Synthetic Data
Initialize the system with realistic traffic patterns.
```bash
python scripts/generate_data.py
```

### 2. Train the Model (Optional)
Retrain the T-GCN model on new data.
```bash
python train.py
```
*Note: A pre-trained model checkpoint is included.*

### 3. Launch the Dashboard
Start the interactive Streamlit command center.
```bash
# Windows
.\run_dashboard.bat

# Linux/Mac
streamlit run app.py
```

---

## 📂 Project Structure

```
traffic-prediction-system/
├── app.py                 # Main Streamlit dashboard application
├── train.py               # Model training script
├── run_dashboard.bat      # Quick launch script for Windows
├── config/                # Configuration files (YAML)
├── data/
│   ├── processed/         # Generated matrices and speed data
├── models/
│   ├── checkpoints/       # Saved model weights (.pth)
│   └── logs/              # Training logs
├── scripts/
│   └── generate_data.py   # Synthetic data generation logic
├── src/
│   ├── models/            # T-GCN model architecture
│   └── utils/             # Routing and graph utilities
└── requirements.txt       # Python dependencies
```

---

## 🧠 Model Architecture (T-GCN)

The system uses a **Temporal Graph Convolutional Network** defined in `src/models/tgcn.py`.

1.  **Graph Convolution (GCN):** Captures the topological structure of the road network (e.g., how traffic at a junction affects connected roads).
2.  **Gated Recurrent Unit (GRU):** Captures temporal dependencies (e.g., how traffic evolves over time from 8:00 AM to 9:00 AM).
3.  **Prediction Layer:** A fully connected layer that outputs the predicted speed for each sensor node.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by the Antigravity Team*
