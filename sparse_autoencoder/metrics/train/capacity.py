"""Capacity Metrics."""

import einops
from jaxtyping import Float
import numpy as np
from numpy import histogram
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
        features: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)],
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT, Axis.BATCH)]:
        r"""Calculate capacities.

        Example:
            >>> import torch
            >>> orthogonal_features = torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])
            >>> orthogonal_caps = CapacityMetric.capacities(orthogonal_features)
            >>> orthogonal_caps
            tensor([[1., 1., 1.]])

        Args:
            features: A collection of features.

        Returns:
            A 1D tensor of capacities, where each element is the capacity of the corresponding
            feature.
        """
        squared_dot_products: Float[Tensor, Axis.names(Axis.BATCH, Axis.BATCH, Axis.COMPONENT)] = (
            einops.einsum(
                features,
                features,
                f"batch_1 {Axis.COMPONENT} {Axis.LEARNT_FEATURE}, \
                    batch_2 {Axis.COMPONENT} {Axis.LEARNT_FEATURE} \
                    -> {Axis.COMPONENT} batch_1 batch_2",
            )
            ** 2
        )

        sum_of_sq_dot: Float[
            Tensor, Axis.names(Axis.COMPONENT, Axis.BATCH)
        ] = squared_dot_products.sum(dim=-1)

        diagonal: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.BATCH)] = torch.diagonal(
            squared_dot_products, dim1=1, dim2=2
        )

        return diagonal / sum_of_sq_dot

    @staticmethod
    def wandb_capacities_histogram(
        capacities: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.BATCH)],
    ) -> list[wandb.Histogram]:
        """Create a W&B histogram of the capacities.

        This can be logged with Weights & Biases using e.g. `wandb.log({"capacities_histogram":
        wandb_capacities_histogram(capacities)})`.

        Args:
            capacities: Capacity of each feature. Can be calculated using :func:`calc_capacities`.

        Returns:
            Weights & Biases histogram for logging with `wandb.log`.
        """
        np_capacities: Float[
            np.ndarray, Axis.names(Axis.COMPONENT, Axis.BATCH)
        ] = capacities.cpu().numpy()

        np_histograms = [histogram(capacity, bins=20, range=(0, 1)) for capacity in np_capacities]

        return [wandb.Histogram(np_histogram=np_histogram) for np_histogram in np_histograms]

    def calculate(self, data: TrainMetricData) -> list[MetricResult]:
        """Calculate the capacities for a training batch."""
        train_batch_capacities = self.capacities(data.learned_activations)

        histograms = self.wandb_capacities_histogram(train_batch_capacities)

        return [
            MetricResult(
                name="capacities",
                component_wise_values=histograms,
                location=self.location,
                aggregate_approach=None,  # Don't aggregate histograms
            )
        ]
