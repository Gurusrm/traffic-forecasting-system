import torch
import unittest
from src.models.st_gnn import STGNN

class TestModels(unittest.TestCase):
    def test_stgnn_initialization(self):
        # Basic smoke test for model init
        config = {
            'model': {
                'num_nodes': 207,
                'in_channels': 2,
                'hidden_channels': 32,
                'out_channels': 12,
                'dropout': 0.3
            }
        }
        try:
            model = STGNN(config['model'])
            self.assertIsInstance(model, torch.nn.Module)
        except Exception as e:
            self.fail(f"STGNN initialization failed: {e}")

if __name__ == '__main__':
    unittest.main()
