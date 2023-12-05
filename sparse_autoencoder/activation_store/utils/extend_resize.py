"""Resize Tensors for Extend Methods."""
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.tensor_types import Axis


def resize_to_list_vectors(
    batched_tensor: Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)],
) -> list[Float[Tensor, Axis.INPUT_OUTPUT_FEATURE]]:
    """Resize Extend List Vectors.

    Takes a tensor of activation vectors, with arbitrary numbers of dimensions (the last of which is
    the neurons dimension), and returns a list of vectors each of size [neurons].

    Examples:
    With 2 axis (e.g. pos neuron):

    >>> import torch
    >>> input = torch.rand(3, 100)
    >>> res = resize_to_list_vectors(input)
    >>> f"{len(res)} items of shape {res[0].shape}"
    '3 items of shape torch.Size([100])'

    With 3 axis (e.g. batch, pos, neuron):

    >>> input = torch.randn(3, 3, 100)
    >>> res = resize_to_list_vectors(input)
    >>> f"{len(res)} items of shape {res[0].shape}"
    '9 items of shape torch.Size([100])'

    With 4 axis (e.g. batch, pos, head_idx, neuron)

    >>> input = torch.rand(3, 3, 3, 100)
    >>> res = resize_to_list_vectors(input)
    >>> f"{len(res)} items of shape {res[0].shape}"
    '27 items of shape torch.Size([100])'

    Args:
        batched_tensor: Input Activation Store Batch

    Returns:
        List of Activation Store Item Vectors
    """
    rearranged: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)] = rearrange(
        batched_tensor,
        "... neurons -> (...) neurons",
    )
    res = rearranged.unbind(0)
    return list(res)


def resize_to_single_item_dimension(
    batch_activations: Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)],
) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]:
    """Resize Extend Single Item Dimension.

    Takes a tensor of activation vectors, with arbitrary numbers of dimensions (the last of which is
    the neurons dimension), and returns a single tensor of size [item, neurons].

    Examples:
    With 2 axis (e.g. pos neuron):

    >>> import torch
    >>> input = torch.rand(3, 100)
    >>> res = resize_to_single_item_dimension(input)
    >>> res.shape
    torch.Size([3, 100])

    With 3 axis (e.g. batch, pos, neuron):

    >>> input = torch.randn(3, 3, 100)
    >>> res = resize_to_single_item_dimension(input)
    >>> res.shape
    torch.Size([9, 100])

    With 4 axis (e.g. batch, pos, head_idx, neuron)

    >>> input = torch.rand(3, 3, 3, 100)
    >>> res = resize_to_single_item_dimension(input)
    >>> res.shape
    torch.Size([27, 100])

    Args:
        batch_activations: Input Activation Store Batch

    Returns:
        Single Tensor of Activation Store Items
    """
    return rearrange(batch_activations, "... neurons -> (...) neurons")
