"""Source model."""
from functools import partial

from einops import rearrange
from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.nn import Module
from transformer_lens import HookedTransformer

from sparse_autoencoder.source_model.store_activations_hook import store_activations_hook
from sparse_autoencoder.tensor_types import Axis


class GenerateActivationsSourceModel(Module):
    """Generate activations source model.

    Wraps a `transformer_lens.HookedTransformer` model so that the forward method returns just the
    required activations. This means that it can be part of a larger module that includes both the
    activation generation and the SAE forward pass (which is useful for parallelization as e.g. DDP
    and Lightning work out of the box if there is just one top-level module).
    """

    cache: list[Int[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]]

    def __init__(
        self, model_name: str, hook_names: list[str], stop_at_layer: int | None = None
    ) -> None:
        """Initialise the source model."""
        super().__init__()
        self.hooked_transformer = HookedTransformer.from_pretrained(model_name)
        self.hook_names = hook_names
        self.stop_at_layer = stop_at_layer
        self.cache = []

        # Add the hooks to store the activations
        hook = partial(store_activations_hook, store=self.cache)
        for hook_name in hook_names:
            self.hooked_transformer.add_hook(hook_name, hook)

        # Disable gradients
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self, input_tokens: Int[Tensor, Axis.names(Axis.BATCH, Axis.POSITION)]
    ) -> Float[Tensor, Axis.names(Axis.STORE_BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)]:
        """Forward pass.

        Args:
            input_tokens: Input tokens.

        Returns:
            Requested activations.
        """
        self.hooked_transformer.forward(
            input=input_tokens, stop_at_layer=self.stop_at_layer, return_type="logits"
        )
        res: Int[
            Tensor, Axis.names(Axis.COMPONENT, Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)
        ] = torch.stack(self.cache)
        self.cache.clear()

        return rearrange(
            res,
            f"{Axis.names(Axis.COMPONENT, Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)} \
                         -> {Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)}",
        )
