"""Capacity Metrics."""

import einops
from jaxtyping import Float
import torch
from torch import Tensor

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
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

    _component_names: list[str]
    full_state_update: bool | None = False
    learned_activations: list[
        Float[Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    ]

    def __init__(self, component_names: list[str]) -> None:
        """Initialize the metric.

        Args:
            component_names: Names of the components.
        """
        super().__init__(component_names, "capacity")
        self._component_names = component_names
        self.add_state(
            "learned_activations",
            default=[],
        )

    def update(
        self,
        input_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> None:
        """Update the metric state.

        Args:
            input_activations: The input activations.
            learned_activations: The learned activations.
            decoded_activations: The decoded activations.
        """
        self.learned_activations.append(learned_activations)

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

    def compute(
        self,
    ) -> dict[str, float | Tensor]:
        """Compute the metric."""
        batch_learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.cat(self.learned_activations)

        capacities: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.BATCH)] = self.capacities(
            batch_learned_activations
        )

        results: dict[str, Float] = dict(zip(self._component_names, capacities))
        results["total"] = capacities.mean(0)
        return results
