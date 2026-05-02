import torch
import unittest
from src.data.dataset import TrafficDataset
from src.data.preprocessing import check_nan_infinite

class TestDataProcessing(unittest.TestCase):
    def test_nan_check(self):
        t = torch.tensor([1.0, float('nan'), 2.0])
        self.assertTrue(torch.isnan(t).any())
        
    def test_dataset_structure(self):
        # Mocking data loading would go here
        pass

if __name__ == '__main__':
    unittest.main()
