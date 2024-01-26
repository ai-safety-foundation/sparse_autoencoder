"""Get the device that the model is on."""
import torch
from torch.nn import Module


def get_model_device(model: Module) -> torch.device:
    """Get the device on which a PyTorch model is on.

    Args:
        model: The PyTorch model.

    Returns:
        The device ('cuda' or 'cpu') where the model is located.

    Raises:
        ValueError: If the model has no parameters.
    """
    # Check if the model has parameters
    if len(list(model.parameters())) == 0:
        exception_message = "The model has no parameters."
        raise ValueError(exception_message)

    # Return the device of the first parameter
    return next(model.parameters()).device
