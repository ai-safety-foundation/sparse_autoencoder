"""L0 norm sparsity metric."""
from jaxtyping import Float, Int64
import torch
from torch import Tensor

from sparse_autoencoder.metrics.train.abstract_train_metric import AbstractTrainMetric
from sparse_autoencoder.tensor_types import Axis


class L0LearnedActivations(AbstractTrainMetric):
    """Learned activations L0 norm sparsity metric.

    The L0 norm is the number of non-zero elements in a learned activation vector. We then average
    this over the batch.

    Example:
        >>> metric = L0LearnedActivations(["mlp1", "mlp2"])
        >>> batch_size = 5
        >>> n_components = 2
        >>> n_input_activations = 3
        >>> n_learned_activations = 6
        >>> input_activations = torch.zeros(batch_size, n_components, n_input_activations)
        >>> learned_activations = torch.ones(batch_size, n_components, n_learned_activations)
        >>> metric.update(input_activations, learned_activations, input_activations)
        >>> metric.compute() # All 6 neurons were active
        {'l0_norm/mlp1': tensor(6.), 'l0_norm/mlp2': tensor(6.), 'l0_norm': tensor(6.)}
    """

    active_neurons_count: Int64[Tensor, Axis.COMPONENT]
    num_activations: Int64[Tensor, Axis.SINGLE_ITEM]

    def __init__(self, component_names: list[str]) -> None:
        """Initialize the metric.

        Args:
            component_names: Names of the components.
        """
        super().__init__(component_names, "l0_norm")
        self._component_names = component_names
        n_components = len(component_names)

        self.add_state(
            "active_neurons_count",
            default=torch.zeros(n_components, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "num_activations",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
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
        self.num_activations += learned_activations.shape[0]

        process_active_neurons_count: Float[Tensor, Axis.COMPONENT] = torch.count_nonzero(
            learned_activations, dim=-1
        ).sum(dim=0, dtype=torch.int64)

        self.active_neurons_count += process_active_neurons_count

    def compute(
        self,
    ) -> dict[str, Tensor]:
        """Compute the metric."""
        batch_activity: Float[Tensor, Axis.COMPONENT] = (
            self.active_neurons_count / self.num_activations
        )
        results = {
            f"{self._metric_name}/{component_name}": value
            for component_name, value in zip(self._component_names, batch_activity)
        }
        results["l0_norm"] = torch.mean(batch_activity)
        return results
