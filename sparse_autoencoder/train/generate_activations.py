"""Generate activations for training a Sparse Autoencoder."""
from functools import partial

import torch
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_store.base_store import (
    ActivationStore,
    StoreFullError,
)
from sparse_autoencoder.src_model.store_activations_hook import store_activations_hook


def generate_activations(
    model: HookedTransformer,
    layer: int,
    cache_name: str,
    store: ActivationStore,
    dataloader: DataLoader,
    num_items: int,
    device: torch.device | None = None,
) -> None:
    """Generate activations for training a Sparse Autoencoder.

    Generates activations and updates the activation store in place.

    Warning:
    This function is a little confusing as it uses side effects. The way it works is to add a hook
    to the model, which will automatically store activations every time the model runs. When it has
    filled up the store to the desired size, it will return `None`. Your activations will then be
    ready in the `store` object that you passed to this function (i.e. it is updated in place). This
    approach is used as it depends on TransformerLens which uses side effects to get the
    activations.

    Args:
        model: The model to generate activations for.
        layer: The layer that you are hooking into to get activations. This is used to stop the
            model at this point rather than generating all remaining activations and logits.
        cache_name: The name of the cache hook point to get activations from. Examples include
            [`hook_embed` `hook_pos_embed`, `blocks.0.hook_resid_pre`, `blocks.0.ln1.hook_scale`,
            `blocks.0.ln1.hook_normalized`, `blocks.0.attn.hook_q`, `blocks.0.attn.hook_k`,
            `blocks.0.attn.hook_v`, `blocks.0.attn.hook_attn_scores`, `blocks.0.attn.hook_pattern`,
            `blocks.0.attn.hook_z`, `blocks.0.hook_attn_out`, `blocks.0.hook_resid_mid`,
            `blocks.0.ln2.hook_scale`, `blocks.0.ln2.hook_normalized`, `blocks.0.mlp.hook_pre`,
            `blocks.0.mlp.hook_post`, `blocks.0.hook_mlp_out`, `blocks.0.hook_resid_post`].
        store: The activation store to use.
        dataloader: Dataloader containing source model input tokens.
        num_items: Number of activation vectors to generate. This is an approximate rather
            than strict limit.
        device: Device to run the model on.
    """
    if isinstance(device, torch.device):
        model.to(device, print_details=False)

    # Add the hook to the model (will automatically store the activations every time the model runs)
    model.remove_all_hook_fns()
    hook = partial(store_activations_hook, store=store)
    model.add_hook(cache_name, hook)

    # Get the input dimensions for logging
    first_item: Int[Tensor, "batch pos"] = next(iter(dataloader))
    batch_size: int = first_item.shape[0]
    context_size: int = first_item.shape[1]
    activations_per_batch: int = context_size * batch_size
    total: int = num_items - num_items % activations_per_batch

    # Loop through the dataloader until the store reaches the desired size
    with torch.no_grad(), tqdm(
        dataloader,
        desc="Generate Activations",
        total=total,
        colour="green",
        position=1,
        leave=False,
        dynamic_ncols=True,
    ) as progress_bar:
        for input_ids in dataloader:
            try:
                input_ids = input_ids.to(device)  # noqa: PLW2901
                model.forward(input_ids, stop_at_layer=layer + 1)  # type: ignore
                progress_bar.update(activations_per_batch)

            # Break the loop if the store is full
            except StoreFullError:
                break

            if len(store) >= total:
                return

        progress_bar.close()
