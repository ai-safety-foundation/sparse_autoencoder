"""L0 norm sparsity metric."""
from typing import TYPE_CHECKING, final

from jaxtyping import Float
import torch
from torch import Tensor

from sparse_autoencoder.metrics.abstract_metric import MetricResult
from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)


if TYPE_CHECKING:
    from sparse_autoencoder.tensor_types import Axis


@final
class TrainBatchLearnedActivationsL0(AbstractTrainMetric):
    """Learned activations L0 norm sparsity metric.

    The L0 norm is the number of non-zero elements in a learned activation vector. We then average
    this over the batch.
    """

    def calculate(self, data: TrainMetricData) -> list[MetricResult]:
        """Create the L0 norm sparsity metric, component wise.."""
        batch_size = data.learned_activations.size(0)

        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = data.learned_activations

        n_non_zero_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT)
        ] = torch.count_nonzero(learned_activations, dim=-1)

        batch_average = n_non_zero_activations / batch_size

        return [
            MetricResult(
                pipeline_location=self.metric_location,
                name="learned_activations_l0_norm",
                component_wise_values=batch_average.tolist(),
                component_names=self._component_names,
            )
        ]
