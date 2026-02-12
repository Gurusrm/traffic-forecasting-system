import torch
from torch.utils.data import Dataset
import numpy as np
import os

class TrafficDataset(Dataset):
    def __init__(self, data, input_window=12, output_window=12):
        """
        Args:
            data (np.ndarray): Shape (num_samples, num_nodes, num_features)
            input_window (int): Number of history steps
            output_window (int): Number of future steps to predict
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        # x: (input_window, num_nodes, num_features)
        # y: (output_window, num_nodes, 1)  (assuming we only predict speed)
        
        x = self.data[idx : idx + self.input_window]
        y = self.data[
            idx + self.input_window : idx + self.input_window + self.output_window
        ]
        
        # If y has more features, we might only want the first one (speed) for loss calc
        # but let's return full for flexibility or slice here.
        # Usually we predict speed.
        if y.shape[-1] > 1:
            y = y[..., :1] 

        return torch.FloatTensor(x), torch.FloatTensor(y)
