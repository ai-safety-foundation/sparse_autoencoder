"""Neuron Resampling."""
from jaxtyping import Bool, Float, Int
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.autoencoder.loss import l1_loss, reconstruction_loss, sae_training_loss
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime


def resample_neurons(
    neuron_activity: Int[Tensor, " learned_features"],
    store: ActivationStore,
    autoencoder: SparseAutoencoder,
    sweep_parameters: SweepParametersRuntime,
    num_inputs: int = 819_200,
) -> None:
    """Resample neurons.

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
        You should reset the Adam optimizer state (to the model parameters) after doing this.

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
    # Get the dead neuron indices
    dead_neuron_indices: Int[Tensor, " learned_neuron_idx"] = torch.where(neuron_activity == 0)[0]

    # Setup stores for loss and inputs
    loss_batches: list[Float[Tensor, " batch_item"]] = []
    input_activations_batches: list[Float[Tensor, "batch_item input_feature"]] = []

    # Compute the loss for the current model on a random subset of 819,200 inputs.
    batch_size: int = sweep_parameters.batch_size
    dataloader = DataLoader(store, batch_size=batch_size)
    batches: int = num_inputs // batch_size

    for batch_idx, batch in enumerate(dataloader):
        input_activations_batches.append(batch)
        learned_activations, reconstructed_activations = autoencoder(batch)

        reconstruction_loss_mse = reconstruction_loss(
            batch,
            reconstructed_activations,
        )
        l1_loss_learned_activations = l1_loss(learned_activations)
        total_loss = sae_training_loss(
            reconstruction_loss_mse,
            l1_loss_learned_activations,
            sweep_parameters.l1_coefficient,
        )
        loss_batches.append(total_loss)

        if batch_idx >= batches:
            break

    # Assign each input vector a probability of being picked that is proportional to the square of
    # the autoencoder's loss on that input.
    square_loss: Float[Tensor, " item"] = torch.stack(loss_batches).pow(2)
    sample_probabilities: Float[Tensor, " item"] = square_loss / square_loss.sum()
    input_activations: Float[Tensor, "item input_feature"] = torch.stack(input_activations_batches)

    encoder_weight: Float[
        Tensor, "learned_feature input_feature"
    ] = autoencoder.encoder_linear.weight
    encoder_bias: Float[Tensor, " learned_feature"] = autoencoder.encoder_linear.bias

    for neuron_idx in dead_neuron_indices:
        # For each dead neuron sample an input according to these probabilities.
        sample_idx: Int[Tensor, 1] = torch.multinomial(sample_probabilities, num_samples=1)[0]
        sampled_input: Float[Tensor, " input_feature"] = input_activations[sample_idx, :]

        # Renormalize the input vector to have unit L2 norm and set this to be the dictionary vector
        # for the dead autoencoder neuron.
        unit_norm_sampled_input: Float[Tensor, " input_feature"] = torch.nn.functional.normalize(
            sampled_input, dim=-1
        )

        # For the corresponding encoder vector, renormalize the input vector to equal the average
        # norm of the encoder weights for alive neurons times 0.2. Set the corresponding encoder
        # bias element to zero.
        alive_neuron_mask: Bool[Tensor, " learned_feature"] = neuron_activity > 0
        alive_encoder_weights: Float[Tensor, "alive_feature input_feature"] = encoder_weight[
            :, alive_neuron_mask
        ]
        average_alive_norm: Float[Tensor, 1] = alive_encoder_weights.norm(dim=-1).mean()
        rescaled_sampled_input: Float[Tensor, " input_feature"] = (
            unit_norm_sampled_input * average_alive_norm * 0.2
        )
        encoder_weight.data[:, neuron_idx] = rescaled_sampled_input
        encoder_bias.data[neuron_idx] = 0.0
