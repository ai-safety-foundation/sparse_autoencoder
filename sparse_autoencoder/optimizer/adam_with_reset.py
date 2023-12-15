"""Adam Optimizer with a reset method.

This reset method is useful when resampling dead neurons during training.
"""
from collections.abc import Iterator
from typing import final

from jaxtyping import Int
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.optimizer import params_t

from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.tensor_types import Axis


@final
class AdamWithReset(Adam, AbstractOptimizerWithReset):
    """Adam Optimizer with a reset method.

    The :meth:`reset_state_all_parameters` and :meth:`reset_neurons_state` methods are useful when
    manually editing the model parameters during training (e.g. when resampling dead neurons). This
    is because Adam maintains running averages of the gradients and the squares of gradients, which
    will be incorrect if the parameters are changed.

    Otherwise this is the same as the standard Adam optimizer.
    """

    parameter_names: list[str]
    """Parameter Names.

    The names of the parameters, so that we can find them later when resetting the state.
    """

    def __init__(  # (extending existing implementation)
        self,
        params: params_t,
        lr: float | Tensor = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        *,
        amsgrad: bool = False,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
        named_parameters: Iterator[tuple[str, Parameter]],
    ) -> None:
        """Initialize the optimizer.

        Warning:
            Named parameters must be with default settings (remove duplicates and not recursive).

        Example:
            >>> import torch
            >>> from sparse_autoencoder.autoencoder.model import SparseAutoencoder
            >>> model = SparseAutoencoder(5, 10, torch.zeros(5))
            >>> optimizer = AdamWithReset(
            ...     model.parameters(),
            ...     named_parameters=model.named_parameters(),
            ... )
            >>> optimizer.reset_state_all_parameters()

        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups.
            lr: Learning rate. A Tensor LR is not yet fully supported for all implementations. Use a
                float LR unless specifying fused=True or capturable=True.
            betas: Coefficients used for computing running averages of gradient and its square.
            eps: Term added to the denominator to improve numerical stability.
            weight_decay: Weight decay (L2 penalty).
            amsgrad: Whether to use the AMSGrad variant of this algorithm from the paper "On the
                Convergence of Adam and Beyond".
            foreach: Whether foreach implementation of optimizer is used. If None, foreach is used
                over the for-loop implementation on CUDA if more performant. Note that foreach uses
                more peak memory.
            maximize: If True, maximizes the parameters based on the objective, instead of
                minimizing.
            capturable: Whether this instance is safe to capture in a CUDA graph. True can impair
                ungraphed performance.
            differentiable: Whether autograd should occur through the optimizer step in training.
                Setting to True can impair performance.
            fused: Whether the fused implementation (CUDA only) is used. Supports torch.float64,
                torch.float32, torch.float16, and torch.bfloat16.
            named_parameters: An iterator over the named parameters of the model. This is used to
                find the parameters when resetting their state. You should set this as
                `model.named_parameters()`.

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
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

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
        neuron_indices: Int[Tensor, Axis.LEARNT_FEATURE_IDX],
        axis: int,
    ) -> None:
        """Reset the state for specific neurons, on a specific parameter.

        Example:
            >>> import torch
            >>> from sparse_autoencoder.autoencoder.model import SparseAutoencoder
            >>> model = SparseAutoencoder(5, 10, torch.zeros(5))
            >>> optimizer = AdamWithReset(
            ...     model.parameters(),
            ...     named_parameters=model.named_parameters(),
            ... )
            >>> # ... train the model and then resample some dead neurons, then do this ...
            >>> dead_neurons_indices = torch.tensor([0, 1]) # Dummy dead neuron indices
            >>> # Reset the optimizer state for parameters that have been updated
            >>> optimizer.reset_neurons_state(model.encoder.weight, dead_neurons_indices, axis=0)
            >>> optimizer.reset_neurons_state(model.encoder.bias, dead_neurons_indices, axis=0)
            >>> optimizer.reset_neurons_state(
            ...     model.decoder.weight,
            ...     dead_neurons_indices,
            ...     axis=1
            ... )

        Args:
            parameter: The parameter to be reset. Examples from the standard sparse autoencoder
                implementation  include `tied_bias`, `_encoder._weight`, `_encoder._bias`,
            neuron_indices: The indices of the neurons to reset.
            axis: The axis of the parameter to reset.
        """
        # Get the state of the parameter
        state = self.state[parameter]

        # Check if state is initialized
        if len(state) == 0:
            return

        # Check there are any neurons to reset
        if neuron_indices.numel() == 0:
            return

        # Reset running averages for the specified neurons
        if "exp_avg" in state:
            exp_avg: Tensor = state["exp_avg"]
            exp_avg.index_fill_(axis, neuron_indices.to(exp_avg.device), 0)
        if "exp_avg_sq" in state:
            exp_avg_sq: Tensor = state["exp_avg_sq"]
            exp_avg_sq.index_fill_(axis, neuron_indices.to(exp_avg_sq.device), 0)

        # If AdamW is used (weight decay fix), also reset the max exp_avg_sq
        if "max_exp_avg_sq" in state:
            max_exp_avg_sq: Tensor = state["max_exp_avg_sq"]
            max_exp_avg_sq.index_fill_(axis, neuron_indices.to(max_exp_avg_sq.device), 0)
