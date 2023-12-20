"""L0 norm sparsity metric."""
from typing import final

import einops
from jaxtyping import Float
import torch
from torch import Tensor

from sparse_autoencoder.metrics.abstract_metric import MetricResult
from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)
from sparse_autoencoder.tensor_types import Axis


@final
class TrainBatchLearnedActivationsL0(AbstractTrainMetric):
    """Learned activations L0 norm sparsity metric.

    The L0 norm is the number of non-zero elements in a learned activation vector. We then average
    this over the batch.
    """

    def calculate(self, data: TrainMetricData) -> list[MetricResult]:
        """Create the L0 norm sparsity metric, component wise.."""
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = data.learned_activations

        n_non_zero_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT)
        ] = torch.count_nonzero(learned_activations, dim=-1).to(dtype=torch.float)

        batch_average: Float[Tensor, Axis.COMPONENT] = einops.reduce(
            n_non_zero_activations, f"{Axis.BATCH} {Axis.COMPONENT} -> {Axis.COMPONENT}", "mean"
        )

        return [
            MetricResult(
                location=self.location,
                name="learned_activations_l0_norm",
                component_wise_values=batch_average,
            )
        ]
