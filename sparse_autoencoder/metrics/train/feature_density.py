"""Train batch feature density."""
import einops
from jaxtyping import Float
import numpy as np
from numpy import histogram
from pydantic import NonNegativeFloat, validate_call
import torch
from torch import Tensor
import wandb

from sparse_autoencoder.metrics.abstract_metric import MetricResult
from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)
from sparse_autoencoder.tensor_types import Axis


class TrainBatchFeatureDensityMetric(AbstractTrainMetric):
    """Train batch feature density.

    Percentage of samples in which each feature was active (i.e. the neuron has "fired"), in a
    training batch.

    Generally we want a small number of features to be active in each batch, so average feature
    density should be low. By contrast if the average feature density is high, it means that the
    features are not sparse enough.

    Warning:
        This is not the same as the feature density of the entire training set. It's main use is
        tracking the progress of training.
    """

    threshold: float

    @validate_call
    def __init__(
        self,
        threshold: NonNegativeFloat = 0.0,
    ) -> None:
        """Initialise the train batch feature density metric.

        Args:
            threshold: Threshold for considering a feature active (i.e. the neuron has "fired").
                This should be close to zero.
        """
        super().__init__()
        self.threshold = threshold

    def feature_density(
        self,
        activations: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)],
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]:
        """Count how many times each feature was active.

        Percentage of samples in which each feature was active (i.e. the neuron has "fired").

        Example:
            >>> import torch
            >>> activations = torch.tensor([[[0.5, 0.5, 0.0]], [[0.5, 0.0, 0.0001]]])
            >>> TrainBatchFeatureDensityMetric(0.001).feature_density(activations).tolist()
            [[1.0, 0.5, 0.0]]

        Args:
            activations: Sample of cached activations (the Autoencoder's learned features).

        Returns:
            Number of times each feature was active in a sample.
        """
        has_fired: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.gt(activations, self.threshold).to(
            dtype=torch.float  # Move to float so it can be averaged
        )

        return einops.reduce(
            has_fired,
            f"{Axis.BATCH} {Axis.COMPONENT} {Axis.LEARNT_FEATURE} \
                -> {Axis.COMPONENT} {Axis.LEARNT_FEATURE}",
            "mean",
        )

    @staticmethod
    def wandb_feature_density_histogram(
        feature_density: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)],
    ) -> list[wandb.Histogram]:
        """Create a W&B histogram of the feature density.

        This can be logged with Weights & Biases using e.g. `wandb.log({"feature_density_histogram":
        wandb_feature_density_histogram(feature_density)})`.

        Args:
            feature_density: Number of times each feature was active in a sample. Can be calculated
                using :func:`feature_activity_count`.

        Returns:
            Weights & Biases histogram for logging with `wandb.log`.
        """
        numpy_feature_density: Float[
            np.ndarray, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = feature_density.cpu().numpy()

        np_histograms = [
            histogram(component_feature_density, bins=50)
            for component_feature_density in numpy_feature_density
        ]

        return [wandb.Histogram(np_histogram=np_histogram) for np_histogram in np_histograms]

    def calculate(self, data: TrainMetricData) -> list[MetricResult]:
        """Calculate the train batch feature density metrics.

        Args:
            data: Train metric data.

        Returns:
            Dictionary with the train batch feature density metric, and a histogram of the feature
            density.
        """
        train_batch_feature_density: Float[
            Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = self.feature_density(data.learned_activations)

        component_wise_histograms = self.wandb_feature_density_histogram(
            train_batch_feature_density
        )

        return [
            MetricResult(
                name="feature_density",
                component_wise_values=component_wise_histograms,
                location=self.location,
                aggregate_approach=None,  # Don't aggregate the histograms
            )
        ]
