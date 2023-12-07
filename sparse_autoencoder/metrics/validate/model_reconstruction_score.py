"""Model reconstruction score."""
from typing import TYPE_CHECKING, Any

from sparse_autoencoder.metrics.validate.abstract_validate_metric import (
    AbstractValidationMetric,
    ValidationMetricData,
)


if TYPE_CHECKING:
    from sparse_autoencoder.tensor_types import Axis
from jaxtyping import Float
from torch import Tensor


class ModelReconstructionScore(AbstractValidationMetric):
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
    """

    def calculate(self, data: ValidationMetricData) -> dict[str, Any]:
        """Calculate the model reconstruction score.

        Example:
            >>> import torch
            >>> data = ValidationMetricData(
            ...     source_model_loss=torch.tensor([2.0, 2.0, 2.0]),
            ...     source_model_loss_with_reconstruction=torch.tensor([3.0, 3.0, 3.0]),
            ...     source_model_loss_with_zero_ablation=torch.tensor([5.0, 5.0, 5.0])
            ... )
            >>> metric = ModelReconstructionScore()
            >>> result = metric.calculate(data)
            >>> round(result['validate/model_reconstruction_score'], 3)
            0.667

        Args:
            data: Validation data.

        Returns:
            Model reconstruction score.
        """
        # Return no statistics if the data is empty (e.g. if we're at the very end of training)
        if data.source_model_loss.numel() == 0:
            return {}

        # Calculate the reconstruction score
        zero_ablate_loss_minus_default_loss: Float[Tensor, Axis.ITEMS] = (
            data.source_model_loss_with_zero_ablation - data.source_model_loss
        )
        zero_ablate_loss_minus_reconstruction_loss: Float[Tensor, Axis.ITEMS] = (
            data.source_model_loss_with_zero_ablation - data.source_model_loss_with_reconstruction
        )
        model_reconstruction_score_itemwise: Float[Tensor, Axis.ITEMS] = (
            zero_ablate_loss_minus_reconstruction_loss / zero_ablate_loss_minus_default_loss
        )
        model_reconstruction_score: float = model_reconstruction_score_itemwise.mean().item()

        # Get the other metrics
        validation_baseline_loss: float = data.source_model_loss.mean().item()
        validation_loss_with_reconstruction: float = (
            data.source_model_loss_with_reconstruction.mean().item()
        )
        validation_loss_with_zero_ablation: float = (
            data.source_model_loss_with_zero_ablation.mean().item()
        )

        return {
            "validate/baseline_loss": validation_baseline_loss,
            "validate/loss_with_reconstruction": validation_loss_with_reconstruction,
            "validate/loss_with_zero_ablation": validation_loss_with_zero_ablation,
            "validate/model_reconstruction_score": model_reconstruction_score,
        }
