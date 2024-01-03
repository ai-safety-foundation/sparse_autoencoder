"""Abstract optimizer with reset."""
from abc import ABC, abstractmethod

from jaxtyping import Int64
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Optimizer

from sparse_autoencoder.tensor_types import Axis


class AbstractOptimizerWithReset(Optimizer, ABC):
    """Abstract optimizer with reset.

    When implementing this interface, we recommend adding a `named_parameters` argument to the
    constructor, which can be obtained from `named_parameters=model.named_parameters()` by the end
    user. This is so that the optimizer can find the parameters to reset.
    """

    @abstractmethod
    def reset_state_all_parameters(self) -> None:
        """Reset the state for all parameters.

        Resets any optimizer state (e.g. momentum). This is for use after manually editing model
            parameters (e.g. with activation resampling).
        """

    @abstractmethod
    def reset_neurons_state(
        self,
        parameter: Parameter,
        neuron_indices: Int64[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX)],
        axis: int,
        component_idx: int,
    ) -> None:
        """Reset the state for specific neurons, on a specific parameter.

        Args:
            parameter: The parameter to reset, e.g. `encoder.Linear.weight`, `encoder.Linear.bias`,
            neuron_indices: The indices of the neurons to reset.
            axis: The axis of the parameter to reset.
            component_idx: The component index of the state values to reset.

        Raises:
            ValueError: If the parameter name is not found.
        """
