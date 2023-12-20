"""Model reconstruction score."""
from typing import TYPE_CHECKING

from sparse_autoencoder.metrics.abstract_metric import MetricResult
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

    def calculate(self, data: ValidationMetricData) -> list[MetricResult]:
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
            >>> round(result[3].aggregate_value.item(), 3)
            0.667

        Args:
            data: Validation data.

        Returns:
            Model reconstruction score.
        """
        # Return no statistics if the data is empty (e.g. if we're at the very end of training)
        if data.source_model_loss.numel() == 0:
            return []

        # Calculate the reconstruction score
        zero_ablate_loss_minus_default_loss: Float[
            Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL)
        ] = data.source_model_loss_with_zero_ablation - data.source_model_loss
        zero_ablate_loss_minus_reconstruction_loss: Float[
            Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL)
        ] = data.source_model_loss_with_zero_ablation - data.source_model_loss_with_reconstruction

        model_reconstruction_score = zero_ablate_loss_minus_reconstruction_loss.mean(
            0
        ) / zero_ablate_loss_minus_default_loss.mean(0)

        # Get the other metrics
        validation_baseline_loss = data.source_model_loss.mean(0)
        validation_loss_with_reconstruction = data.source_model_loss_with_reconstruction.mean(0)
        validation_loss_with_zero_ablation = data.source_model_loss_with_zero_ablation.mean(0)

        return [
            MetricResult(
                component_wise_values=validation_baseline_loss,
                location=self.location,
                name="reconstruction_score",
                postfix="baseline_loss",
            ),
            MetricResult(
                component_wise_values=validation_loss_with_reconstruction,
                location=self.location,
                name="reconstruction_score",
                postfix="loss_with_reconstruction",
            ),
            MetricResult(
                component_wise_values=validation_loss_with_zero_ablation,
                location=self.location,
                name="reconstruction_score",
                postfix="loss_with_zero_ablation",
            ),
            MetricResult(
                component_wise_values=model_reconstruction_score,
                location=self.location,
                name="reconstruction_score",
            ),
        ]
