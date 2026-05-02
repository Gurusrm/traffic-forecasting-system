import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_prediction(target, prediction, output_path=None):
    """
    Plot target vs prediction for a single node
    """
    plt.figure(figsize=(12, 6))
    plt.plot(target, label='Target')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    plt.title("Traffic Speed Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Speed")
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_heatmap(data, output_path=None):
    """
    Plot heatmap of traffic speed across nodes and time
    data: (Time, Nodes)
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.T, cmap='viridis')
    plt.title("Traffic Speed Heatmap")
    plt.xlabel("Time Step")
    plt.ylabel("Node ID")
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
