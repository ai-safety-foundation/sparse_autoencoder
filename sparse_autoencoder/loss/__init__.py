"""Loss Modules.

Loss modules are specialised PyTorch modules that calculate the loss for a Sparse Autoencoder. They
all inherit from AbstractLoss, which defines the interface for loss modules and some common methods.

If you want to create your own loss function, see :class:`AbstractLoss`.

For combining multiple loss modules into a single loss module, see :class:`LossReducer`.
"""
from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossLogType, LossReductionType
from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss
from sparse_autoencoder.loss.mse_reconstruction_loss import MSEReconstructionLoss
from sparse_autoencoder.loss.reducer import LossReducer


__all__ = [
    "AbstractLoss",
    "LearnedActivationsL1Loss",
    "LossLogType",
    "LossReducer",
    "LossReductionType",
    "MSEReconstructionLoss",
]
