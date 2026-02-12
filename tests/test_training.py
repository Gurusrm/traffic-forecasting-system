import unittest
import torch

class TestTraining(unittest.TestCase):
    def test_optimizer_setup(self):
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.assertEqual(optimizer.defaults['lr'], 0.001)

if __name__ == '__main__':
    unittest.main()
