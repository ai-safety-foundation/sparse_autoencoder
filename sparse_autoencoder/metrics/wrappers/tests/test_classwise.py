"""Test the classwise wrapper."""
import pytest
import torch

from sparse_autoencoder.metrics.train.feature_density import FeatureDensityMetric
from sparse_autoencoder.metrics.train.l0_norm import L0NormMetric
from sparse_autoencoder.metrics.wrappers.classwise import ClasswiseWrapperWithMean


@pytest.mark.parametrize(
    ("num_components"),
    [
        pytest.param(1, id="Single component"),
        pytest.param(2, id="Multiple components"),
    ],
)
def test_feature_density_classwise_wrapper(num_components: int) -> None:
    """Test the classwise wrapper."""
    metric = FeatureDensityMetric(3, num_components)
    component_names = [f"mlp_{n}" for n in range(num_components)]
    wrapped_metric = ClasswiseWrapperWithMean(metric, component_names, prefix="feature_density")

    learned_activations = torch.tensor([[[1.0, 1.0, 1.0]] * num_components])
    expected_output = torch.tensor([[1.0, 1.0, 1.0]])
    res = wrapped_metric.forward(learned_activations=learned_activations)

    for component in component_names:
        assert torch.allclose(res[f"feature_density/{component}"], expected_output)

    assert torch.allclose(res["feature_density/mean"], expected_output.mean(0))


@pytest.mark.parametrize(
    ("num_components"),
    [
        pytest.param(1, id="Single component"),
        pytest.param(2, id="Multiple components"),
    ],
)
def test_l0_norm_classwise_wrapper(num_components: int) -> None:
    """Test the classwise wrapper."""
    metric = L0NormMetric(num_components)
    component_names = [f"mlp_{n}" for n in range(num_components)]
    wrapped_metric = ClasswiseWrapperWithMean(metric, component_names, prefix="l0")

    learned_activations = torch.tensor([[[1.0, 0.0, 1.0]] * num_components])
    expected_output = torch.tensor([2.0])
    res = wrapped_metric.forward(learned_activations=learned_activations)

    for component in component_names:
        assert torch.allclose(res[f"l0/{component}"], expected_output)

    assert torch.allclose(res["l0/mean"], expected_output.mean(0))
