"""Methods to reshape activation tensors."""
from collections.abc import Callable
from functools import reduce
from typing import TypeAlias

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.tensor_types import Axis


ReshapeActivationsFunction: TypeAlias = Callable[
    [Float[Tensor, Axis.names(Axis.ANY)]],
    Float[Tensor, Axis.names(Axis.STORE_BATCH, Axis.INPUT_OUTPUT_FEATURE)],
]
"""Reshape Activations Function.

Used within hooks to e.g. reshape activations before storing them in the activation store.
"""


def reshape_to_last_dimension(
    batch_activations: Float[Tensor, Axis.names(Axis.ANY)],
) -> Float[Tensor, Axis.names(Axis.STORE_BATCH, Axis.INPUT_OUTPUT_FEATURE)]:
    """Reshape to Last Dimension.

    Takes a tensor of activation vectors, with arbitrary numbers of dimensions (the last of which is
    the neurons dimension), and returns a single tensor of size [item, neurons].

    Examples:
        With 2 axis (e.g. pos neuron):

        >>> import torch
        >>> input = torch.rand(3, 100)
        >>> res = reshape_to_last_dimension(input)
        >>> res.shape
        torch.Size([3, 100])

        With 3 axis (e.g. batch, pos, neuron):

        >>> input = torch.randn(3, 3, 100)
        >>> res = reshape_to_last_dimension(input)
        >>> res.shape
        torch.Size([9, 100])

        With 4 axis (e.g. batch, pos, head_idx, neuron)

        >>> input = torch.rand(3, 3, 3, 100)
        >>> res = reshape_to_last_dimension(input)
        >>> res.shape
        torch.Size([27, 100])

    Args:
        batch_activations: Input Activation Store Batch

    Returns:
        Single Tensor of Activation Store Items
    """
    return rearrange(batch_activations, "... input_output_feature -> (...) input_output_feature")


def reshape_concat_last_dimensions(
    batch_activations: Float[Tensor, Axis.names(Axis.ANY)],
    concat_dims: int,
) -> Float[Tensor, Axis.names(Axis.STORE_BATCH, Axis.INPUT_OUTPUT_FEATURE)]:
    """Reshape to Last Dimension, Concatenating the Specified Dimensions.

    Takes a tensor of activation vectors, with arbitrary numbers of dimensions (the last
    `concat_dims` of which are the neuron dimensions), and returns a single tensor of size
    [item, neurons].

    Examples:
        With 3 axis (e.g. batch, pos, neuron), concatenating the last 2 dimensions:

        >>> import torch
        >>> input = torch.randn(3, 4, 5)
        >>> res = reshape_concat_last_dimensions(input, 2)
        >>> res.shape
        torch.Size([3, 20])

        With 4 axis (e.g. batch, pos, head_idx, neuron), concatenating the last 3 dimensions:

        >>> input = torch.rand(2, 3, 4, 5)
        >>> res = reshape_concat_last_dimensions(input, 3)
        >>> res.shape
        torch.Size([2, 60])

    Args:
        batch_activations: Input Activation Store Batch
        concat_dims: Number of dimensions to concatenate

    Returns:
        Single Tensor of Activation Store Items
    """
    neurons = reduce(lambda x, y: x * y, batch_activations.shape[-concat_dims:])
    items = reduce(lambda x, y: x * y, batch_activations.shape[:-concat_dims])

    return batch_activations.reshape(items, neurons)
