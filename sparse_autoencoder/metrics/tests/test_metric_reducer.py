"""Tests for the metric reducer class(es)."""
from collections import OrderedDict
from typing import Any

import pytest
import torch

from sparse_autoencoder.metrics.abstract_metric import AbstractTrainMetric, TrainMetricData
from sparse_autoencoder.metrics.reducer import TrainingMetricReducer


@pytest.fixture()
def metric1() -> "AbstractTrainMetric":
    """Create a metric that always returns 1."""
    class Metric1(AbstractTrainMetric):
        def create_progress_bar_postfix(self, data: TrainMetricData) -> OrderedDict[str, Any]:
            raise NotImplementedError
        def create_weights_and_biases_log(self, data: TrainMetricData) -> OrderedDict[str, Any]:
            return OrderedDict(metric1=1)
    return Metric1()

@pytest.fixture()
def metric2() -> "AbstractTrainMetric":
    """Create a metric that always returns 2."""
    class Metric2(AbstractTrainMetric):
        def create_progress_bar_postfix(self, data: TrainMetricData) -> OrderedDict[str, Any]:
            raise NotImplementedError
        def create_weights_and_biases_log(self, data: TrainMetricData) -> OrderedDict[str, Any]:
            return OrderedDict(metric2=2)
    return Metric2()

@pytest.fixture()
def metric_reducer(metric1: "AbstractTrainMetric",
                   metric2: "AbstractTrainMetric") -> TrainingMetricReducer:
    """Create a metric reducer."""
    return TrainingMetricReducer(metric1, metric2)

@pytest.fixture()
def train_metric_data() -> TrainMetricData:
    """Create some train metric data."""
    return TrainMetricData(
        input_activations=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]),
        learned_activations=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]),
        decoded_activations=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]),
    )

def test_create_weights_and_biases_log(metric_reducer: TrainingMetricReducer,
                                       train_metric_data: TrainMetricData) -> None:
    """Test the create_weights_and_biases_log method."""
    log = metric_reducer.create_weights_and_biases_log(train_metric_data)
    assert log == OrderedDict(metric1=1, metric2=2)
