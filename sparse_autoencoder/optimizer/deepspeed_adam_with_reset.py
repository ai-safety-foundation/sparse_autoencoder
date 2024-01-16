"""Deepspeed Zero One Adam Optimizer with a reset method.

This reset method is useful when resampling dead neurons during training.
"""
from collections.abc import Iterator
from typing import final

from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from jaxtyping import Int
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim.optimizer import params_t

from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.tensor_types import Axis


@final
class ZeroOneAdamWithReset(ZeroOneAdam, AbstractOptimizerWithReset):
    """Deepspeed Zero One Adam Optimizer with a reset method.

    https://deepspeed.readthedocs.io/en/latest/optimizers.html#zerooneadam-gpu

    The :meth:`reset_state_all_parameters` and :meth:`reset_neurons_state` methods are useful when
    manually editing the model parameters during training (e.g. when resampling dead neurons). This
    is because Adam maintains running averages of the gradients and the squares of gradients, which
    will be incorrect if the parameters are changed.

    Otherwise this is the same as the standard ZeroOneAdam optimizer.

    Warning:
        Requires a distributed torch backend.
    """

    parameter_names: list[str]
    """Parameter Names.

    The names of the parameters, so that we can find them later when resetting the state.
    """

    _has_components_dim: bool
    """Whether the parameters have a components dimension."""

    def __init__(
        self,
        params: params_t,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        named_parameters: Iterator[tuple[str, Parameter]],
        has_components_dim: bool,
    ) -> None:
        """Initialize the optimizer.

        Warning:
            Named parameters must be with default settings (remove duplicates and not recursive).

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate. A Tensor LR is not yet fully supported for all implementations. Use a
                float LR unless specifying fused=True or capturable=True.
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            named_parameters: An iterator over the named parameters of the model. This is used to
                find the parameters when resetting their state. You should set this as
                `model.named_parameters()`.
            has_components_dim: If the parameters have a components dimension (i.e. if you are
                training an SAE on more than one component).


        Raises:
            ValueError: If the number of parameter names does not match the number of parameters.
        """
        # Initialise the parent class (note we repeat the parameter names so that type hints work).
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        self._has_components_dim = has_components_dim

        # Store the names of the parameters, so that we can find them later when resetting the
        # state.
        self.parameter_names = [name for name, _value in named_parameters]

        if len(self.parameter_names) != len(self.param_groups[0]["params"]):
            error_message = (
                "The number of parameter names does not match the number of parameters. "
                "If using model.named_parameters() make sure remove_duplicates is True "
                "and recursive is False (the default settings)."
            )
            raise ValueError(error_message)

    def reset_state_all_parameters(self) -> None:
        """Reset the state for all parameters.

        Iterates over all parameters and resets both the running averages of the gradients and the
        squares of gradients.
        """
        # Iterate over every parameter
        for group in self.param_groups:
            for parameter in group["params"]:
                # Get the state
                state = self.state[parameter]

                # Check if state is initialized
                if len(state) == 0:
                    continue

                # Reset running averages
                exp_avg: Tensor = state["exp_avg"]
                exp_avg.zero_()
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                exp_avg_sq.zero_()

                # If AdamW is used (weight decay fix), also reset the max exp_avg_sq
                if "max_exp_avg_sq" in state:
                    max_exp_avg_sq: Tensor = state["max_exp_avg_sq"]
                    max_exp_avg_sq.zero_()

    def reset_neurons_state(
        self,
        parameter: Parameter,
        neuron_indices: Int[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)],
        axis: int,
        component_idx: int = 0,
    ) -> None:
        """Reset the state for specific neurons, on a specific parameter.

        Args:
            parameter: The parameter to be reset. Examples from the standard sparse autoencoder
                implementation  include `tied_bias`, `_encoder._weight`, `_encoder._bias`,
            neuron_indices: The indices of the neurons to reset.
            axis: The axis of the state values to reset (i.e. the input/output features axis, as
                we're resetting all input/output features for a specific dead neuron).
            component_idx: The component index of the state values to reset.

        Raises:
            ValueError: If the parameter has a components dimension, but has_components_dim is
                False.
        """
        # Get the state of the parameter
        state = self.state[parameter]

        # If the number of dimensions is 3, we definitely have a components dimension. If 2, we may
        # do (as the bias has 2 dimensions with components, but the weight has 2 dimensions without
        # components).
        definitely_has_components_dimension = 3
        if (
            not self._has_components_dim
            and state["exp_avg"].ndim == definitely_has_components_dimension
        ):
            error_message = (
                "The parameter has a components dimension, but has_components_dim is False. "
                "This should not happen."
            )
            raise ValueError(error_message)

        # Check if state is initialized
        if len(state) == 0:
            return

        # Check there are any neurons to reset
        if neuron_indices.numel() == 0:
            return

        # Move the neuron indices to the correct device
        neuron_indices = neuron_indices.to(device=state["exp_avg"].device)

        # Reset running averages for the specified neurons
        if "exp_avg" in state:
            if self._has_components_dim:
                state["exp_avg"][component_idx].index_fill_(axis, neuron_indices, 0)
            else:
                state["exp_avg"].index_fill_(axis, neuron_indices, 0)

        if "exp_avg_sq" in state:
            if self._has_components_dim:
                state["exp_avg_sq"][component_idx].index_fill_(axis, neuron_indices, 0)
            else:
                state["exp_avg_sq"].index_fill_(axis, neuron_indices, 0)

        # If AdamW is used (weight decay fix), also reset the max exp_avg_sq
        if "max_exp_avg_sq" in state:
            if self._has_components_dim:
                state["max_exp_avg_sq"][component_idx].index_fill_(axis, neuron_indices, 0)
            else:
                state["max_exp_avg_sq"].index_fill_(axis, neuron_indices, 0)
