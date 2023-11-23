"""Loss reducer."""
from collections.abc import Iterator
from typing import final

from jaxtyping import Float
import torch
from torch import Tensor

from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    ItemTensor,
    LearnedActivationBatch,
)


@final
class LossReducer(AbstractLoss):
    """Loss reducer.

    Reduces multiple loss algorithms into a single loss algorithm (by summing). Analogous to
    nn.Sequential.

    Example:
        >>> from sparse_autoencoder.loss.mse_reconstruction_loss import MSEReconstructionLoss
        >>> from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss
        >>> LossReducer(
        ...     MSEReconstructionLoss(),
        ...     LearnedActivationsL1Loss(0.001),
        ... )
        LossReducer(
          (0): MSEReconstructionLoss()
          (1): LearnedActivationsL1Loss(l1_coefficient=0.001)
        )

    """

    _modules: dict[str, "AbstractLoss"]
    """Children loss modules."""

    def __init__(
        self,
        *loss_modules: AbstractLoss,
    ):
        """Initialize the loss reducer.

        Args:
            *loss_modules: Loss modules to reduce.

        Raises:
            ValueError: If the loss reducer has no loss modules.
        """
        super().__init__()

        for idx, loss_module in enumerate(loss_modules):
            self._modules[str(idx)] = loss_module

        if len(self) == 0:
            error_message = "Loss reducer must have at least one loss module."
            raise ValueError(error_message)

    def forward(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,
    ) -> ItemTensor:
        """Reduce loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Mean loss across the batch, summed across the loss modules.
        """
        all_modules_loss: Float[Tensor, "module train_batch"] = torch.stack(
            [
                loss_module.forward(source_activations, learned_activations, decoded_activations)
                for loss_module in self._modules.values()
            ]
        )

        return all_modules_loss.sum(dim=0)

    def __dir__(self) -> list[str]:
        """Dir dunder method."""
        return list(self._modules.__dir__())

    def __getitem__(self, idx: int) -> AbstractLoss:
        """Get item dunder method."""
        return self._modules[str(idx)]

    def __iter__(self) -> Iterator[AbstractLoss]:
        """Iterator dunder method."""
        return iter(self._modules.values())

    def __len__(self) -> int:
        """Length dunder method."""
        return len(self._modules)
