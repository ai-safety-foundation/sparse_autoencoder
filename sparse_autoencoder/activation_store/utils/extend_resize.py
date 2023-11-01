"""Resize Tensors for Extend Methods."""
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.activation_store.base_store import (
    ActivationStoreBatch,
    ActivationStoreItem,
)


def resize_to_list_vectors(
    input: ActivationStoreBatch,
) -> list[ActivationStoreItem]:
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
        input: Input Activation Store Batch

    Returns:
        List of Activation Store Item Vectors
    """
    rearranged: Float[Tensor, "batch neuron"] = rearrange(
        input, "... neurons -> (...) neurons"
    )
    res = rearranged.unbind(0)
    return list(res)


def resize_to_single_item_dimension(
    input: ActivationStoreBatch,
) -> Float[Tensor, "item neuron"]:
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
        input: Input Activation Store Batch

    Returns:
        Single Tensor of Activation Store Items
    """
    return rearrange(input, "... neurons -> (...) neurons")
