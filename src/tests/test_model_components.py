import torch
import unittest
from src.models.correlation_module import CorrelationModule
from src.models.identification_module import IdentificationModule

class TestModelComponents(unittest.TestCase):
    def test_correlation_module(self):
        dummy_frame = torch.randn(3, 224, 224)
        corr_module = CorrelationModule()
        output = corr_module(dummy_frame, dummy_frame, dummy_frame)
        self.assertEqual(output.shape, dummy_frame.shape, "Output shape mismatch!")

    def test_identification_module(self):
        dummy_input = torch.randn(1, 3, 16, 224, 224)
        identification_module = IdentificationModule(input_channels=3)
        output = identification_module(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape, "Output shape mismatch!")

if __name__ == '__main__':
    unittest.main()
