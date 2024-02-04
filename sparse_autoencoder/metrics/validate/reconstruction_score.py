"""Reconstruction score metric."""
from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis


class ReconstructionScoreMetric(Metric):
    r"""Model reconstruction score.

    Creates a score that measures how well the model can reconstruct the data.

    $$
    \begin{align*}
        v &= \text{number of validation items} \\
        l \in{\mathbb{R}^v} &= \text{loss with no changes to the source model} \\
        l_\text{recon} \in{\mathbb{R}^v} &= \text{loss with reconstruction} \\
        l_\text{zero} \in{\mathbb{R}^v} &= \text{loss with zero ablation} \\
        s &= \text{reconstruction score} \\
        s_\text{itemwise} &= \frac{l_\text{zero} - l_\text{recon}}{l_\text{zero} - l} \\
        s &= \sum_{i=1}^v s_\text{itemwise} / v
    \end{align*}
    $$

    Example:
        >>> metric = ReconstructionScoreMetric(num_components=1)
        >>> source_model_loss=torch.tensor([2.0, 2.0, 2.0])
        >>> source_model_loss_with_reconstruction=torch.tensor([3.0, 3.0, 3.0])
        >>> source_model_loss_with_zero_ablation=torch.tensor([5.0, 5.0, 5.0])
        >>> metric.forward(
        ...     source_model_loss=source_model_loss,
        ...     source_model_loss_with_reconstruction=source_model_loss_with_reconstruction,
        ...     source_model_loss_with_zero_ablation=source_model_loss_with_zero_ablation
        ... )
        tensor(0.6667)
    """

    # Torchmetrics settings
    is_differentiable: bool | None = False
    full_state_update: bool | None = True

    # State
    source_model_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL]
    source_model_loss_with_zero_ablation: Float[Tensor, Axis.COMPONENT_OPTIONAL]
    source_model_loss_with_reconstruction: Float[Tensor, Axis.COMPONENT_OPTIONAL]
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(self, num_components: PositiveInt = 1) -> None:
        """Initialise the metric."""
        super().__init__()

        self.add_state(
            "source_model_loss", default=torch.zeros(num_components), dist_reduce_fx="sum"
        )
        self.add_state(
            "source_model_loss_with_zero_ablation",
            default=torch.zeros(num_components),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "source_model_loss_with_reconstruction",
            default=torch.zeros(num_components),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        source_model_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL],
        source_model_loss_with_reconstruction: Float[Tensor, Axis.COMPONENT_OPTIONAL],
        source_model_loss_with_zero_ablation: Float[Tensor, Axis.COMPONENT_OPTIONAL],
        component_idx: int = 0,
    ) -> None:
        """Update the metric state.

        Args:
            source_model_loss: Loss with no changes to the source model.
            source_model_loss_with_reconstruction: Loss with SAE reconstruction.
            source_model_loss_with_zero_ablation: Loss with zero ablation.
            component_idx: Component idx.
        """
        self.source_model_loss[component_idx] += source_model_loss.sum()
        self.source_model_loss_with_zero_ablation[
            component_idx
        ] += source_model_loss_with_zero_ablation.sum()
        self.source_model_loss_with_reconstruction[
            component_idx
        ] += source_model_loss_with_reconstruction.sum()

    def compute(
        self,
    ) -> Float[Tensor, Axis.COMPONENT_OPTIONAL]:
        """Compute the metric."""
        zero_ablate_loss_minus_reconstruction_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL] = (
            self.source_model_loss_with_zero_ablation - self.source_model_loss_with_reconstruction
        )

        zero_ablate_loss_minus_default_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL] = (
            self.source_model_loss_with_zero_ablation - self.source_model_loss
        )

        return zero_ablate_loss_minus_reconstruction_loss / zero_ablate_loss_minus_default_loss
