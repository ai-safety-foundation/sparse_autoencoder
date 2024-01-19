# """Tests for the NeuronActivityMetric class."""
# from jaxtyping import Float, Int64
# import pytest
# import torch
# from torch import Tensor

# from sparse_autoencoder.metrics.train.neuron_activity_metric import (
#     NeuronActivityMetric,
# )
# from sparse_autoencoder.metrics.utils.find_metric_result import find_metric_result
# from sparse_autoencoder.tensor_types import Axis


# class TestNeuronActivityMetric:
#     """Test the NeuronActivityMetric class."""

#     @pytest.mark.parametrize(
#         ("learned_activations", "expected_dead_count", "expected_alive_count"),
#         [
#             pytest.param(
#                 torch.tensor([[[0.0, 0, 0, 0, 0]]]),
#                 torch.tensor([5]),
#                 torch.tensor([0]),
#                 id="All dead",
#             ),
#             pytest.param(
#                 torch.tensor([[[1.0, 1, 1, 1, 1]]]),
#                 torch.tensor([0]),
#                 torch.tensor([5]),
#                 id="All alive",
#             ),
#             pytest.param(
#                 torch.tensor([[[0.0, 1, 0, 1, 0]]]),
#                 torch.tensor([3]),
#                 torch.tensor([2]),
#                 id="Some dead",
#             ),
#             pytest.param(
#                 torch.tensor([[[0.0, 1, 0, 1, 0], [0.0, 0, 0, 0, 0]]]),
#                 torch.tensor([3, 5]),
#                 torch.tensor([2, 0]),
#                 id="Multiple components with some dead",
#             ),
#         ],
#     )
#     def test_dead_neuron_count(
#         self,
#         learned_activations: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)],
#         expected_dead_count: Int64[Tensor, Axis.names(Axis.COMPONENT)],
#         expected_alive_count: Int64[Tensor, Axis.names(Axis.COMPONENT)],
#     ) -> None:
#         """Test if dead neuron count is correctly calculated."""
#         input_activations = torch.zeros_like(learned_activations, dtype=torch.float)
#         data = TrainMetricData(
#             learned_activations=learned_activations,
#             # Input and decoded activations are not used in this metric
#             input_activations=input_activations,
#             decoded_activations=input_activations,
#         )
#         neuron_activity_metric = NeuronActivityMetric(horizons=[1])
#         metrics = neuron_activity_metric.calculate(data)

#         dead_over_1_activations = find_metric_result(metrics, postfix="dead_over_1_activations")
#         alive_over_1_activations = find_metric_result(metrics, postfix="alive_over_1_activations")

#         assert isinstance(dead_over_1_activations.component_wise_values, torch.Tensor)
#         assert isinstance(alive_over_1_activations.component_wise_values, torch.Tensor)
#         assert torch.allclose(dead_over_1_activations.component_wise_values, expected_dead_count)
#         assert torch.allclose(alive_over_1_activations.component_wise_values, expected_alive_count)
