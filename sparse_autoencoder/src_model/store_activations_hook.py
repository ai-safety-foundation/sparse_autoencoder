from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.activation_store.base_store import ActivationStore


def store_activations_hook(
    value: Float[Tensor, "*batch_and_pos neuron"],
    hook: HookPoint,
    store: ActivationStore,
):
    pass
