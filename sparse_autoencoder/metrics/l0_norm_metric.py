"""L0 (sparsity) norm metric."""
from typing import Any, final
from collections import OrderedDict

import torch

from sparse_autoencoder.metrics.abstract_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)

@final
class L0NormMetric(AbstractTrainMetric):
    """L0 (sparsity) norm metric."""
    @final
    def create_progress_bar_postfix(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @final
    def create_weights_and_biases_log(self, data: TrainMetricData) -> OrderedDict[str, float]:
        """Create a log item for Weights and Biases."""
        # The L0 norm is the number of non-zero elements
        # (We're averaging over the batch)
        acts = data.learned_activations
        value = (torch.sum(acts != 0) / acts.size(0)).item()
        return OrderedDict(l0_norm=value)
