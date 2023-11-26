"""Zero ablate hook."""
import torch
from torch import Tensor
from transformer_lens.hook_points import HookPoint


def zero_ablate_hook(
    value: Tensor,
    hook: HookPoint,  # noqa: ARG001
) -> Tensor:
    """Zero ablate hook.

    Args:
        value: The activations to store.
        hook: The hook point.

    Example:
        >>> dummy_hook_point = HookPoint()
        >>> value = torch.ones(2, 3)
        >>> zero_ablate_hook(value, dummy_hook_point)
        tensor([[0., 0., 0.],
                [0., 0., 0.]])

    Returns:
        Replaced activations.
    """
    return torch.zeros_like(value)
