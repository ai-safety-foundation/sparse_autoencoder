"""Capacity Metrics."""
from typing import Any

import einops
from jaxtyping import Float
import numpy as np
from numpy import histogram
from numpy.typing import NDArray
import torch
from torch import Tensor
import wandb

from sparse_autoencoder.metrics.abstract_metric import MetricResult
from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)
from sparse_autoencoder.tensor_types import Axis


class CapacityMetric(AbstractTrainMetric):
    """Capacities Metrics for Learned Features.

    Measure the capacity of a set of features as defined in [Polysemanticity and Capacity in Neural
    Networks](https://arxiv.org/pdf/2210.01892.pdf).

    Capacity is intuitively measuring the 'proportion of a dimension' assigned to a feature.
    Formally it's the ratio of the squared dot product of a feature with itself to the sum of its
    squared dot products of all features.

    If the features are orthogonal, the capacity is 1. If they are all the same, the capacity is
    1/n.
    """

    @staticmethod
    def capacities(
        features: Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)],
    ) -> Float[Tensor, Axis.BATCH]:
        r"""Calculate capacities.

        $$
        \text{squared_dot_products} =

        $$

        Example:
            >>> import torch
            >>> orthogonal_features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
            >>> orthogonal_caps = CapacityMetric.capacities(orthogonal_features)
            >>> orthogonal_caps
            tensor([1., 1., 1.])

        Args:
            features: A collection of features.

        Returns:
            A 1D tensor of capacities, where each element is the capacity of the corresponding
            feature.
        """
        squared_dot_products = (
            einops.einsum(
                features,
                features,
                "batch_1 feat_dim, batch_2 feat_dim \
                    -> batch_1 batch_2",
            )
            ** 2
        )

        sum_of_sq_dot: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT)
        ] = squared_dot_products.sum(dim=-1)

        return torch.diag(squared_dot_products) / sum_of_sq_dot

    @staticmethod
    def wandb_capacities_histogram(
        capacities: Float[Tensor, Axis.BATCH],
    ) -> wandb.Histogram:
        """Create a W&B histogram of the capacities.

        This can be logged with Weights & Biases using e.g. `wandb.log({"capacities_histogram":
        wandb_capacities_histogram(capacities)})`.

        Args:
            capacities: Capacity of each feature. Can be calculated using :func:`calc_capacities`.

        Returns:
            Weights & Biases histogram for logging with `wandb.log`.
        """
        numpy_capacities: NDArray[np.float_] = capacities.detach().cpu().numpy()

        bins, values = histogram(numpy_capacities, bins=20, range=(0, 1))
        return wandb.Histogram(np_histogram=(bins, values))

    def calculate(self, data: TrainMetricData) -> list[MetricResult]:
        """Calculate the capacities for a training batch."""
        histograms = []

        for component_learned_activation in data.learned_activations:
            train_batch_capacities = self.capacities(component_learned_activation)
            histograms.append(self.wandb_capacities_histogram(train_batch_capacities))

        return [
            MetricResult(
                name="capacities",
                component_wise_values=histograms,
                pipeline_location=self.metric_location,
                component_names=self._component_names,
            )
        ]
