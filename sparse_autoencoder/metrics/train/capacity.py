"""Capacity Metrics."""
from typing import Any

import einops
from jaxtyping import Float
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis


class CapacityMetric(Metric):
    """Capacities metric.

    Measure the capacity of a set of features as defined in [Polysemanticity and Capacity in Neural
    Networks](https://arxiv.org/pdf/2210.01892.pdf).

    Capacity is intuitively measuring the 'proportion of a dimension' assigned to a feature.
    Formally it's the ratio of the squared dot product of a feature with itself to the sum of its
    squared dot products of all features.

    Warning:
        This is memory intensive as it requires caching all learned activations for a batch.

    Examples:
        If the features are orthogonal, the capacity is 1.

        >>> metric = CapacityMetric()
        >>> learned_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [1., 0., 1.] # Component 1: learned features
        ...     ],
        ...     [ # Batch 2
        ...         [0., 1., 0.] # Component 1: learned features (orthogonal)
        ...     ]
        ... ])
        >>> metric.forward(learned_activations)
        tensor([[1., 1.]])

        If they are all the same, the capacity is 1/n.

        >>> learned_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [1., 1., 1.] # Component 1: learned features
        ...     ],
        ...     [ # Batch 2
        ...         [1., 1., 1.] # Component 1: learned features (same)
        ...     ]
        ... ])
        >>> metric.forward(learned_activations)
        tensor([[0.5000, 0.5000]])
    """

    # Torchmetrics settings
    is_differentiable: bool | None = False
    full_state_update: bool | None = False
    plot_lower_bound: float | None = 0.0
    plot_upper_bound: float | None = 1.0

    # State
    learned_activations: list[
        Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]
    ]

    def __init__(self) -> None:
        """Initialize the metric."""
        super().__init__()
        self.add_state("learned_activations", default=[])

    def update(
        self,
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        **kwargs: Any,  # type: ignore # noqa: ARG002, ANN401 (allows combining with other metrics)
    ) -> None:
        """Update the metric state.

        Args:
            learned_activations: The learned activations.
            **kwargs: Ignored keyword arguments (to allow use with other metrics in a collection).
        """
        self.learned_activations.append(learned_activations)

    @staticmethod
    def capacities(
        features: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.BATCH)]:
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
        squared_dot_products: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.BATCH, Axis.COMPONENT_OPTIONAL)
        ] = (
            einops.einsum(
                features,
                features,
                f"batch_1 ... {Axis.LEARNT_FEATURE}, \
                    batch_2 ... {Axis.LEARNT_FEATURE} \
                    -> ... batch_1 batch_2",
            )
            ** 2
        )

        sum_of_sq_dot: Float[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.BATCH)
        ] = squared_dot_products.sum(dim=-1)

        diagonal: Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.BATCH)] = torch.diagonal(
            squared_dot_products, dim1=1, dim2=2
        )

        return diagonal / sum_of_sq_dot

    def compute(
        self,
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.BATCH)]:
        """Compute the metric."""
        batch_learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ] = torch.cat(self.learned_activations)

        return self.capacities(batch_learned_activations)
