import numpy as np
import pickle

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def load_data(dataset_dir, batch_size=64, valid_batch_size=None, test_batch_size=None):
    # This is a placeholder for the actual data loading logic
    # In a real scenario, we would load .npz files here
    pass

def generate_train_val_test(data, train_ratio=0.7, val_ratio=0.1):
    """
    Split data into train, val, test
    data: (num_samples, num_nodes, num_features)
    """
    num_samples = len(data)
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data
