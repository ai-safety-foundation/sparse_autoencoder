"""L0 norm sparsity metric."""
from typing import final

import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)


@final
class TrainBatchLearnedActivationsL0(AbstractTrainMetric):
    """Learned activations L0 norm sparsity metric.

    The L0 norm is the number of non-zero elements in a learned activation vector. We then average
    this over the batch.
    """

    def calculate(self, data: TrainMetricData) -> dict[str, float]:
        """Create a log item for Weights and Biases."""
        batch_size = data.learned_activations.size(0)
        n_non_zero_activations = torch.count_nonzero(data.learned_activations)
        batch_average = n_non_zero_activations / batch_size
        return {"train/learned_activations_l0_norm": batch_average.item()}
