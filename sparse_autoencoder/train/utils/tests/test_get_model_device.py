"""Test get_model_device.py."""
import pytest
import torch
from torch import Tensor
from torch.nn import Linear, Module

from sparse_autoencoder.train.utils.get_model_device import get_model_device


def test_model_on_cpu() -> None:
    """Test that it returns CPU when the model is on CPU."""

    class TestModel(Module):
        """Test model."""

        def __init__(self) -> None:
            """Initialize the model."""
            super().__init__()
            self.fc = Linear(10, 5)

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass."""
            return self.fc(x)

    model = TestModel()
    model.to("cpu")
    assert get_model_device(model) == torch.device("cpu"), "Device should be CPU"


# Test with a model that has no parameters
def test_model_no_parameters() -> None:
    """Test that it raises a ValueError when the model has no parameters."""

    class EmptyModel(Module):
        def forward(self, x: Tensor) -> Tensor:
            return x

    model = EmptyModel()
    with pytest.raises(ValueError, match="The model has no parameters."):
        _ = get_model_device(model)
