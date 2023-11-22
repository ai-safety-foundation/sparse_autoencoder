"""Tests for the L0NormMetric class."""
import numpy as np
import pytest
import torch

from sparse_autoencoder.metrics.histogram_metric import HistogramMetric
from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData


@pytest.fixture()
def histogram_metric() -> HistogramMetric:
    """Create a histogram metric."""
    return HistogramMetric()


@pytest.fixture()
def train_metric_data() -> TrainMetricData:
    """Create some train metric data."""
    return TrainMetricData(
        input_activations=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        learned_activations=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        decoded_activations=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_create_weights_and_biases_log(
    histogram_metric: HistogramMetric, train_metric_data: TrainMetricData
) -> None:
    """Test the create_weights_and_biases_log method."""
    log = histogram_metric.calculate(train_metric_data)

    expected_hist, _ = np.histogram([0.0, 1 / 3, 2 / 3], bins=64)
    assert (log["histogram"].histogram == expected_hist).all()
    assert log["num_dead_features"] == 1
