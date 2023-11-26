"""Test the zero ablate hook."""
import pytest
import torch
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.source_model.zero_ablate_hook import zero_ablate_hook


class MockHookPoint(HookPoint):
    """Mock HookPoint class."""


@pytest.fixture()
def mock_hook_point() -> MockHookPoint:
    """Fixture to provide a mock HookPoint instance."""
    return MockHookPoint()


def test_zero_ablate_hook_with_standard_tensor(mock_hook_point: MockHookPoint) -> None:
    """Test zero_ablate_hook with a standard tensor.

    Args:
        mock_hook_point: A mock HookPoint instance.
    """
    value = torch.ones(3, 4)
    expected = torch.zeros(3, 4)
    result = zero_ablate_hook(value, mock_hook_point)
    assert torch.equal(result, expected), "The output tensor should contain only zeros."


@pytest.mark.parametrize("shape", [(10,), (5, 5), (2, 3, 4)])
def test_zero_ablate_hook_with_various_shapes(
    mock_hook_point: MockHookPoint, shape: tuple[int, ...]
) -> None:
    """Test zero_ablate_hook with tensors of various shapes.

    Args:
        mock_hook_point: A mock HookPoint instance.
        shape: A tuple representing the shape of the tensor.
    """
    value = torch.ones(*shape)
    expected = torch.zeros(*shape)
    result = zero_ablate_hook(value, mock_hook_point)
    assert torch.equal(
        result, expected
    ), f"The output tensor should be of shape {shape} with zeros."


def test_float_dtype_maintained(mock_hook_point: MockHookPoint) -> None:
    """Test that the float dtype is maintained.

    Args:
        mock_hook_point: A mock HookPoint instance.
    """
    value = torch.ones(3, 4, dtype=torch.float)
    result = zero_ablate_hook(value, mock_hook_point)
    assert result.dtype == torch.float, "The output tensor should be of dtype float."
