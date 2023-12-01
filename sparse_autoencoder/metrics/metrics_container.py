"""Metrics container."""
from dataclasses import dataclass, field

from sparse_autoencoder.metrics.generate.abstract_generate_metric import AbstractGenerateMetric
from sparse_autoencoder.metrics.train.abstract_train_metric import AbstractTrainMetric
from sparse_autoencoder.metrics.train.capacity import CapacityMetric
from sparse_autoencoder.metrics.train.feature_density import TrainBatchFeatureDensityMetric
from sparse_autoencoder.metrics.train.l0_norm_metric import TrainBatchLearnedActivationsL0
from sparse_autoencoder.metrics.train.neuron_activity_metric import NeuronActivityMetric
from sparse_autoencoder.metrics.validate.abstract_validate_metric import AbstractValidationMetric
from sparse_autoencoder.metrics.validate.model_reconstruction_score import ModelReconstructionScore


@dataclass
class MetricsContainer:
    """Metrics container.

    Stores all metrics used in a pipeline.
    """

    generate_metrics: list[AbstractGenerateMetric] = field(default_factory=list)
    """Metrics for the generate section."""

    train_metrics: list[AbstractTrainMetric] = field(default_factory=list)
    """Metrics for the train section."""

    validation_metrics: list[AbstractValidationMetric] = field(default_factory=list)
    """Metrics for the validation section."""


default_metrics = MetricsContainer(
    train_metrics=[
        TrainBatchFeatureDensityMetric(),
        CapacityMetric(),
        TrainBatchLearnedActivationsL0(),
        NeuronActivityMetric(),
    ],
    validation_metrics=[ModelReconstructionScore()],
)
"""Default metrics container."""
