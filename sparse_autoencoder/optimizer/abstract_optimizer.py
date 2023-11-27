"""Abstract optimizer with reset."""
from abc import ABC, abstractmethod

from torch.optim import Optimizer

from sparse_autoencoder.tensor_types import LearntNeuronIndices


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
        parameter_name: str,
        neuron_indices: LearntNeuronIndices,
        axis: int,
        parameter_group: int = 0,
    ) -> None:
        """Reset the state for specific neurons, on a specific parameter.

        Args:
            parameter_name: The name of the parameter. Examples from the standard sparse autoencoder
                implementation  include `tied_bias`, `encoder.Linear.weight`, `encoder.Linear.bias`,
                `decoder.Linear.weight`, and `decoder.ConstrainedUnitNormLinear.weight`.
            neuron_indices: The indices of the neurons to reset.
            axis: The axis of the parameter to reset.
            parameter_group: The index of the parameter group to reset (typically this is just zero,
                unless you have setup multiple parameter groups for e.g. different learning rates
                for different parameters).

        Raises:
            ValueError: If the parameter name is not found.
        """
