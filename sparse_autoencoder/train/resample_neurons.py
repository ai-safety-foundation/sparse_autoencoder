"""Neuron Resampling."""

from typing import TYPE_CHECKING

from einops import rearrange
from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.autoencoder.loss import l1_loss, reconstruction_loss, sae_training_loss
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime


if TYPE_CHECKING:
    from sparse_autoencoder.autoencoder.components.unit_norm_linear import ConstrainedUnitNormLinear


def get_dead_neuron_indices(
    neuron_activity: Int[Tensor, " learned_features"], threshold: int = 0
) -> Int[Tensor, " dead_neuron"]:
    """Identify the indices of neurons that have zero activity.

    Example:
        >>> neuron_activity = torch.tensor([0, 0, 3, 10, 0])
        >>> dead_neuron_indices = get_dead_neuron_indices(neuron_activity)
        >>> dead_neuron_indices.tolist()
        [0, 1, 4]

    Args:
        neuron_activity: Tensor representing the number of times each neuron fired.
        threshold: Threshold for determining if a neuron is dead (has fired less than this number of
            times.

    Returns:
        A tensor containing the indices of neurons that are 'dead' (zero activity).
    """
    return torch.where(neuron_activity <= threshold)[0]


def compute_loss_and_get_activations(
    store: ActivationStore,
    autoencoder: SparseAutoencoder,
    sweep_parameters: SweepParametersRuntime,
    num_inputs: int,
) -> tuple[Float[Tensor, " item"], Float[Tensor, "item input_feature"]]:
    """Compute the loss on a random subset of inputs.

    Computes the loss and also stores the input activations (for use in resampling neurons).

    Args:
        store: Activation store.
        autoencoder: Sparse autoencoder model.
        sweep_parameters: Current training sweep parameters.
        num_inputs: Number of input activations to use.

    Returns:
        A tuple containing the loss per item, and all input activations.
    """
    with torch.no_grad():
        loss_batches: list[Float[Tensor, " batch_item"]] = []
        input_activations_batches: list[Float[Tensor, "batch_item input_feature"]] = []
        batch_size: int = sweep_parameters.batch_size
        dataloader = DataLoader(store, batch_size=batch_size)
        batches: int = num_inputs // batch_size

        for batch_idx, batch in enumerate(iter(dataloader)):
            input_activations_batches.append(batch)
            learned_activations, reconstructed_activations = autoencoder(batch)
            loss_batches.append(
                sae_training_loss(
                    reconstruction_loss(batch, reconstructed_activations),
                    l1_loss(learned_activations),
                    sweep_parameters.l1_coefficient,
                )
            )
            if batch_idx >= batches:
                break

        loss = torch.cat(loss_batches)
        input_activations = torch.cat(input_activations_batches)

        # Check we generated enough data
        if len(loss) < num_inputs:
            error_message = (
                f"Cannot get {num_inputs} items from the store, as only {len(loss)} were available."
            )
            raise ValueError(error_message)

        return loss, input_activations


def assign_sampling_probabilities(loss: Float[Tensor, " item"]) -> Tensor:
    """Assign the sampling probabilities for each input activations vector.

    Assign each input vector a probability of being picked that is proportional to the square of
    the autoencoder's loss on that input.

    Example:
        >>> loss = torch.tensor([1.0, 2.0, 3.0])
        >>> assign_sampling_probabilities(loss).round(decimals=1)
        tensor([0.1000, 0.3000, 0.6000])

    Args:
        loss: Loss per item.

    Returns:
        A tensor of probabilities for each item.
    """
    square_loss = loss.pow(2)
    return square_loss / square_loss.sum()


def sample_input(
    probabilities: Float[Tensor, " item"],
    input_activations: Float[Tensor, "item input_feature"],
    num_samples: int,
) -> Float[Tensor, "dead_neuron input_feature"]:
    """Sample an input vector based on the provided probabilities.

    Example:
        >>> probabilities = torch.tensor([0.1, 0.2, 0.7])
        >>> input_activations = torch.tensor([[1, 2], [3, 4], [5, 6]])
        >>> _seed = torch.manual_seed(0)  # For reproducibility in example
        >>> sampled_input = sample_input(probabilities, input_activations, 2)
        >>> sampled_input.tolist()
        [[5, 6], [3, 4]]

    Args:
        probabilities: Probabilities for each input.
        input_activations: Input activation vectors.
        num_samples: Number of samples to take (number of dead neurons).

    Returns:
        Sampled input activation vector.

    Raises:
        ValueError: If the number of samples is greater than the number of input activations.
    """
    if num_samples > len(input_activations):
        exception_message = (
            f"Cannot sample {num_samples} inputs from {len(input_activations)} input activations."
        )
        raise ValueError(exception_message)

    if num_samples == 0:
        return torch.empty(
            (0, input_activations.shape[-1]),
            dtype=input_activations.dtype,
            device=input_activations.device,
        )

    sample_indices: Int[Tensor, " dead_neuron"] = torch.multinomial(
        probabilities, num_samples=num_samples
    )
    return input_activations[sample_indices, :]


def renormalize_and_scale(
    sampled_input: Float[Tensor, "dead_neuron input_feature"],
    neuron_activity: Int[Tensor, " learned_features"],
    encoder_weight: Float[Tensor, "learned_feature input_feature"],
) -> Float[Tensor, "dead_neuron input_feature"]:
    """Renormalize and scale the resampled dictionary vectors.

    Renormalize the input vector to equal the average norm of the encoder weights for alive neurons
    times 0.2.

    Example:
        >>> _seed = torch.manual_seed(0)  # For reproducibility in example
        >>> sampled_input = torch.tensor([[3.0, 4.0]])
        >>> neuron_activity = torch.tensor([3, 0, 5, 0, 1, 3])
        >>> encoder_weight = torch.ones((6, 2))
        >>> rescaled_input = renormalize_and_scale(sampled_input, neuron_activity, encoder_weight)
        >>> rescaled_input.round(decimals=1)
        tensor([[0.2000, 0.2000]])

    Args:
        sampled_input: Tensor of the sampled input activation.
        neuron_activity: Tensor representing the number of times each neuron fired.
        encoder_weight: Tensor of encoder weights.

    Returns:
        Rescaled sampled input.

    Raises:
        ValueError: If there are no alive neurons.
    """
    alive_neuron_mask: Bool[Tensor, " learned_features"] = neuron_activity > 0

    # Check there is at least one alive neuron
    if not torch.any(alive_neuron_mask):
        error_message = "No alive neurons found."
        raise ValueError(error_message)

    # Handle all alive neurons
    if torch.all(alive_neuron_mask):
        return torch.empty(
            (0, sampled_input.shape[-1]), dtype=sampled_input.dtype, device=sampled_input.device
        )

    # Calculate the average norm of the encoder weights for alive neurons.
    alive_encoder_weights: Float[Tensor, "learned_feature alive_input_features"] = encoder_weight[
        alive_neuron_mask, :
    ]
    average_alive_norm: Float[Tensor, 1] = alive_encoder_weights.norm(dim=-1).mean()

    # Renormalize the input vector to equal the average norm of the encoder weights for alive
    # neurons times 0.2.
    renormalized_input: Float[Tensor, "dead_neuron input_feature"] = torch.nn.functional.normalize(
        sampled_input, dim=-1
    )
    return renormalized_input * (average_alive_norm * 0.2)


def resample_dead_neurons(
    neuron_activity: Int[Tensor, " learned_features"],
    store: ActivationStore,
    autoencoder: SparseAutoencoder,
    sweep_parameters: SweepParametersRuntime,
    num_inputs: int = 819_200,
) -> None:
    """Resample dead neurons.

    Over the course of training, a subset of autoencoder neurons will have zero activity across a
    large number of datapoints. The authors of *Towards Monosemanticity: Decomposing Language Models
    With Dictionary Learning* found that “resampling” these dead neurons during training improves
    the number of likely-interpretable features (i.e., those in the high density cluster) and
    reduces total loss. This resampling may be compatible with the Lottery Ticket Hypothesis and
    increase the number of chances the network has to find promising feature directions.

    An interesting nuance around dead neurons involves the ultralow density cluster. They found that
    if we increase the number of training steps then networks will kill off more of these ultralow
    density neurons. This reinforces the use of the high density cluster as a useful metric because
    there can exist neurons that are de facto dead but will not appear to be when looking at the
    number of dead neurons alone.

    This approach is designed to seed new features to fit inputs where the current autoencoder
    performs worst. Resetting the encoder norm and bias are crucial to ensuring this resampled
    neuron will only fire weakly for inputs similar to the one used for its reinitialization. This
    was done to minimize interference with the rest of the network.

    Warning:
        The optimizer should be reset after applying this function, as the Adam state will be
        incorrect for the modified weights and biases.

        Note this approach is also known to create sudden loss spikes, and resampling too frequently
        causes training to diverge.

    Args:
        neuron_activity: Number of times each neuron fired.
        store: Activation store.
        autoencoder: Sparse autoencoder model.
        sweep_parameters: Current training sweep parameters.
        num_inputs: Number of input activations to use when resampling. Will be rounded down to be
            divisible by the batch size, and cannot be larger than the number of items currently in
            the store.
    """
    with torch.no_grad():
        dead_neuron_indices = get_dead_neuron_indices(neuron_activity)

        # Compute the loss for the current model on a random subset of inputs and get the
        # activations.
        loss, input_activations = compute_loss_and_get_activations(
            store, autoencoder, sweep_parameters, num_inputs
        )

        # Assign each input vector a probability of being picked that is proportional to the square
        # of the autoencoder's loss on that input.
        sample_probabilities: Float[Tensor, " item"] = assign_sampling_probabilities(loss)

        # Get references to the encoder and decoder parameters
        encoder_linear: torch.nn.Linear = autoencoder.encoder.get_submodule("Linear")  # type: ignore
        decoder_linear: ConstrainedUnitNormLinear = autoencoder.decoder.get_submodule(
            "ConstrainedUnitNormLinear"
        )  # type: ignore
        encoder_weight: Float[Tensor, "learned_feature input_feature"] = encoder_linear.weight
        encoder_bias: Float[Tensor, " learned_feature"] = encoder_linear.bias
        decoder_weight: Float[Tensor, "input_feature learned_feature"] = decoder_linear.weight

        # For each dead neuron sample an input according to these probabilities.
        sampled_input: Float[Tensor, "dead_neuron input_feature"] = sample_input(
            sample_probabilities, input_activations, len(dead_neuron_indices)
        )

        # Renormalize the input vector to have unit L2 norm and set this to be the dictionary
        # vector for the dead autoencoder neuron.
        renormalized_input: Float[
            Tensor, "dead_neuron input_feature"
        ] = torch.nn.functional.normalize(sampled_input, dim=-1)

        decoder_weight[:, dead_neuron_indices] = rearrange(
            renormalized_input, "dead_neuron input_feature -> input_feature dead_neuron"
        )

        # For the corresponding encoder vector, renormalize the input vector to equal the
        # average norm of the encoder weights for alive neurons times 0.2. Set the corresponding
        # encoder bias element to zero.
        rescaled_sampled_input = renormalize_and_scale(
            sampled_input, neuron_activity, encoder_weight
        )
        encoder_weight.data[dead_neuron_indices, :] = rescaled_sampled_input
        encoder_bias.data[dead_neuron_indices] = 0.0
