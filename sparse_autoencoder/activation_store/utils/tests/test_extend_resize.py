"""Tests of the Extend Resize Functions."""
import pytest
import torch

from sparse_autoencoder.activation_store.utils.extend_resize import (
    resize_to_list_vectors,
    resize_to_single_item_dimension,
)
from sparse_autoencoder.tensor_types import InputOutputActivationBatch


class TestResizeListVectors:
    """Resize to List Vectors Tests."""

    @pytest.mark.parametrize(
        ("input_shape", "expected_len", "expected_shape"),
        [
            ((3, 100), 3, torch.Size([100])),
            ((3, 3, 100), 9, torch.Size([100])),
            ((3, 3, 3, 100), 27, torch.Size([100])),
        ],
    )
    def test_resize_to_list_vectors(
        self,
        input_shape: tuple[int],
        expected_len: int,
        expected_shape: torch.Tensor,
    ) -> None:
        """Check each item's shape in the resulting list."""
        input_tensor = torch.rand(input_shape)
        result = resize_to_list_vectors(InputOutputActivationBatch(input_tensor))

        assert len(result) == expected_len, f"Expected list of length {expected_len}"
        assert all(
            item.shape == expected_shape for item in result
        ), f"All items should have shape {expected_shape}"

    def test_resize_to_list_vectors_values(self) -> None:
        """Check each item's values in the resulting list."""
        input_tensor = torch.tensor([[[1.0, 2], [3, 4]], [[5, 6], [7, 8]]])
        expected_output = [
            torch.tensor([1.0, 2]),
            torch.tensor([3.0, 4]),
            torch.tensor([5.0, 6]),
            torch.tensor([7.0, 8]),
        ]
        result = resize_to_list_vectors(InputOutputActivationBatch(input_tensor))

        for expected, output in zip(expected_output, result, strict=True):
            assert torch.all(
                torch.eq(expected, output),
            ), "Tensor values do not match expected"


class TestResizeSingleItemDimension:
    """Resize to Single Item Dimension Tests."""

    @pytest.mark.parametrize(
        ("input_shape", "expected_shape"),
        [
            ((3, 100), (3, 100)),
            ((3, 3, 100), (9, 100)),
            ((3, 3, 3, 100), (27, 100)),
        ],
    )
    def test_resize_to_single_item_dimension(
        self,
        input_shape: tuple[int],
        expected_shape: tuple[int],
    ) -> None:
        """Check the resulting tensor shape."""
        input_tensor = torch.randn(input_shape)
        result = resize_to_single_item_dimension(InputOutputActivationBatch(input_tensor))

        assert result.shape == expected_shape, f"Expected tensor shape {expected_shape}"

    def test_resize_to_single_item_dimension_values(self) -> None:
        """Check the resulting tensor values."""
        input_tensor = torch.tensor([[[1.0, 2], [3, 4]], [[5, 6], [7, 8]]])
        expected_output = torch.tensor([[1.0, 2], [3, 4], [5, 6], [7, 8]])
        result = resize_to_single_item_dimension(InputOutputActivationBatch(input_tensor))

        assert torch.all(
            torch.eq(expected_output, result),
        ), "Tensor values do not match expected"
