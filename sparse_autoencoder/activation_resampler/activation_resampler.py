"""Activation resampler."""
from typing import Annotated

from einops import rearrange
from jaxtyping import Bool, Float, Int64
from pydantic import Field, NonNegativeInt, PositiveInt, validate_call
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
    ParameterUpdateResults,
)
from sparse_autoencoder.activation_resampler.utils.component_slice_tensor import (
    get_component_slice_tensor,
)
from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.train.utils import get_model_device


class ActivationResampler(AbstractActivationResampler):
    """Activation resampler.

    Collates the number of times each neuron fires over a set number of learned activation vectors,
    and then provides the parameters necessary to reset any dead neurons.

    Motivation:
        Over the course of training, a subset of autoencoder neurons will have zero activity across
        a large number of datapoints. The authors of *Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning* found that “resampling” these dead neurons during training
        improves the number of likely-interpretable features (i.e., those in the high density
        cluster) and reduces total loss. This resampling may be compatible with the Lottery Ticket
        Hypothesis and increase the number of chances the network has to find promising feature
        directions.

        An interesting nuance around dead neurons involves the ultralow density cluster. They found
        that if we increase the number of training steps then networks will kill off more of these
        ultralow density neurons. This reinforces the use of the high density cluster as a useful
        metric because there can exist neurons that are de facto dead but will not appear to be when
        looking at the number of dead neurons alone.

        This approach is designed to seed new features to fit inputs where the current autoencoder
        performs worst. Resetting the encoder norm and bias are crucial to ensuring this resampled
        neuron will only fire weakly for inputs similar to the one used for its reinitialization.
        This was done to minimize interference with the rest of the network.

    Warning:
        The optimizer should be reset after applying this function, as the Adam state will be
        incorrect for the modified weights and biases.

    Warning:
        This approach is also known to create sudden loss spikes, and resampling too frequently
        causes training to diverge.
    """

    _activations_seen_since_last_resample: int = 0
    """Number of activations since we last resampled."""

    _collated_neuron_activity: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    """Collated neuron activity, over the current data collection window."""

    _threshold_is_dead_portion_fires: float
    """Threshold for determining if a neuron has fired (or is dead)."""

    _max_n_resamples: int
    """Maximum number of times that resampling should be performed."""

    _n_activations_collated_since_last_resample: int = 0
    """Number of activations collated since we last resampled.

    Number of vectors used to collate neuron activity, over the current collation window.
    """

    _n_components: int
    """Number of components."""

    _n_times_resampled: int = 0
    """Number of times that resampling has been performed."""

    neuron_activity_window_end: int
    """End of the window for collecting neuron activity."""

    neuron_activity_window_start: int
    """Start of the window for collecting neuron activity."""

    @validate_call
    def __init__(
        self,
        n_learned_features: PositiveInt,
        n_components: NonNegativeInt = 1,
        resample_interval: PositiveInt = 200_000_000,
        max_n_resamples: NonNegativeInt = 4,
        n_activations_activity_collate: PositiveInt = 100_000_000,
        resample_dataset_size: PositiveInt = 819_200,
        threshold_is_dead_portion_fires: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.0,
    ) -> None:
        r"""Initialize the activation resampler.

        Defaults to values used in the Anthropic Towards Monosemanticity paper.

        Args:
            n_learned_features: Number of learned features
            n_components: Number of components that the SAE is being trained on.
            resample_interval: Interval in number of autoencoder input activation vectors trained
                on, before resampling.
            max_n_resamples: Maximum number of resamples to perform throughout the entire pipeline.
                Set to inf if you want to have no limit.
            n_activations_activity_collate: Number of autoencoder learned activation vectors to
                collate before resampling (the activation resampler will start collecting on vector
                $\text{resample_interval} - \text{n_steps_collate}$).
            resample_dataset_size: Number of autoencoder input activations to use for calculating
                the loss, as part of the resampling process to create the reset neuron weights.
            threshold_is_dead_portion_fires: Threshold for determining if a neuron is dead (has
                "fired" in less than this portion of the collated sample).

        Raises:
            ValueError: If any of the arguments are invalid (e.g. negative integers).
        """
        if n_activations_activity_collate > resample_interval:
            error_message = (
                "Number of steps to collate must be less than or equal to the resample interval."
            )
            raise ValueError(error_message)

        super().__init__()
        self.neuron_activity_window_end = resample_interval
        self.neuron_activity_window_start = resample_interval - n_activations_activity_collate
        self._max_n_resamples = max_n_resamples
        self._collated_neuron_activity = torch.zeros(
            (n_components, n_learned_features), dtype=torch.int64
        )
        self._resample_dataset_size = resample_dataset_size
        self._threshold_is_dead_portion_fires = threshold_is_dead_portion_fires
        self._n_components = n_components

    def _get_dead_neuron_indices(
        self,
    ) -> list[Int64[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)]]:
        """Identify the indices of neurons that are dead.

        Identifies any neurons that have fired less than the threshold portion of the collated
        sample size.

        Example:
            >>> resampler = ActivationResampler(n_learned_features=6, n_components=2)
            >>> resampler._collated_neuron_activity = torch.tensor(
            ...     [[1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 0]]
            ... )
            >>> resampler._get_dead_neuron_indices()
            [tensor([2, 3]), tensor([5])]

        Returns:
            List of dead neuron indices for each component.

        Raises:
            ValueError: If no neuron activity has been collated yet.
        """
        # Check we have already collated some neuron activity
        if torch.all(self._collated_neuron_activity == 0):
            error_message = "Cannot get dead neuron indices without neuron activity."
            raise ValueError(error_message)

        # Find any neurons that fire less than the threshold portion of times
        threshold_is_dead_n_fires: int = int(
            self._n_activations_collated_since_last_resample * self._threshold_is_dead_portion_fires
        )

        return [
            torch.where(self._collated_neuron_activity[component_idx] <= threshold_is_dead_n_fires)[
                0
            ].to(dtype=torch.int64)
            for component_idx in range(self._n_components)
        ]

    def compute_loss_and_get_activations(
        self,
        store: ActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> tuple[
        Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)],
        Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)],
    ]:
        """Compute the loss on a random subset of inputs.

        Motivation:
            Helps find input vectors that have high SAE loss, so that we can resample dead neurons
            in a way that improves performance on these specific input vectors.

        Args:
            store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).

        Returns:
            A tuple of loss per item, and all input activations.

        Raises:
            ValueError: If the number of items in the store is less than the number of inputs
        """
        with torch.no_grad():
            loss_batches: list[Float[Tensor, Axis.BATCH]] = []
            input_activations_batches: list[
                Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]
            ] = []
            dataloader = DataLoader(store, batch_size=train_batch_size)
            n_inputs = self._resample_dataset_size
            n_batches_required: int = n_inputs // train_batch_size
            model_device: torch.device = get_model_device(autoencoder)

            for batch_idx, batch in enumerate(iter(dataloader)):
                input_activations_batches.append(batch)
                source_activations = batch.to(model_device)
                learned_activations, reconstructed_activations = autoencoder(source_activations)
                loss_batches.append(
                    loss_fn.forward(
                        source_activations, learned_activations, reconstructed_activations
                    )
                )
                if batch_idx >= n_batches_required:
                    break

            loss_result = torch.cat(loss_batches).to(model_device)
            input_activations = torch.cat(input_activations_batches).to(model_device)

            # Check we generated enough data
            if len(loss_result) < n_inputs:
                error_message = (
                    f"Cannot get {n_inputs} items from the store, "
                    f"as only {len(loss_result)} were available."
                )
                raise ValueError(error_message)

            return loss_result, input_activations

    @staticmethod
    def assign_sampling_probabilities(
        loss: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]:
        """Assign the sampling probabilities for each input activations vector.

        Assign each input vector a probability of being picked that is proportional to the square of
        the autoencoder's loss on that input.

        Examples:
            >>> loss = torch.tensor([1.0, 2.0, 3.0])
            >>> ActivationResampler.assign_sampling_probabilities(loss).round(decimals=2)
            tensor([0.0700, 0.2900, 0.6400])

            >>> loss = torch.tensor([[1.0, 2], [2, 4], [3, 6]])
            >>> ActivationResampler.assign_sampling_probabilities(loss).round(decimals=2)
            tensor([[0.0700, 0.0700],
                    [0.2900, 0.2900],
                    [0.6400, 0.6400]])

        Args:
            loss: Loss per item.

        Returns:
            A tensor of probabilities for each item.
        """
        square_loss = loss.pow(2)
        return square_loss / square_loss.sum(0)

    @staticmethod
    def sample_input(
        probabilities: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)],
        input_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        n_samples: list[int],
    ) -> list[Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]]:
        """Sample an input vector based on the provided probabilities.

        Example:
            >>> probabilities = torch.tensor([[0.1], [0.2], [0.7]])
            >>> input_activations = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
            >>> _seed = torch.manual_seed(0)  # For reproducibility in example
            >>> sampled_input = ActivationResampler.sample_input(
            ...     probabilities, input_activations, [2]
            ... )
            >>> sampled_input[0].tolist()
            [[5.0, 6.0], [3.0, 4.0]]

        Args:
            probabilities: Probabilities for each input.
            input_activations: Input activation vectors.
            n_samples: Number of samples to take (number of dead neurons).

        Returns:
            Sampled input activation vector.

        Raises:
            ValueError: If the number of samples is greater than the number of input activations.
        """
        sampled_inputs: list[
            Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]
        ] = []

        for component_idx, component_n_samples in enumerate(n_samples):
            component_probabilities: Float[Tensor, Axis.BATCH] = get_component_slice_tensor(
                input_tensor=probabilities,
                n_dim_with_component=2,
                component_dim=1,
                component_idx=component_idx,
            )

            component_input_activations: Float[
                Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)
            ] = get_component_slice_tensor(
                input_tensor=input_activations,
                n_dim_with_component=3,
                component_dim=1,
                component_idx=component_idx,
            )

            if component_n_samples > len(component_input_activations):
                exception_message = (
                    f"Cannot sample {component_n_samples} inputs from "
                    f"{len(component_input_activations)} input activations."
                )
                raise ValueError(exception_message)

            # Handle the 0 dead neurons case
            if component_n_samples == 0:
                sampled_inputs.append(
                    torch.empty(
                        (0, component_input_activations.shape[-1]),
                        dtype=component_input_activations.dtype,
                        device=component_input_activations.device,
                    )
                )
                continue

            # Handle the 1+ dead neuron case
            component_sample_indices: Int64[Tensor, Axis.LEARNT_FEATURE_IDX] = torch.multinomial(
                component_probabilities, num_samples=component_n_samples
            )
            sampled_inputs.append(component_input_activations[component_sample_indices, :])

        return sampled_inputs

    @staticmethod
    def renormalize_and_scale(
        sampled_input: Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)],
        neuron_activity: Int64[Tensor, Axis.names(Axis.LEARNT_FEATURE)],
        encoder_weight: Float[Tensor, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)],
    ) -> Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]:
        """Renormalize and scale the resampled dictionary vectors.

        Renormalize the input vector to equal the average norm of the encoder weights for alive
        neurons times 0.2.

        Example:
            >>> from torch.nn import Parameter
            >>> _seed = torch.manual_seed(0)  # For reproducibility in example
            >>> sampled_input = torch.tensor([[3.0, 4.0]])
            >>> neuron_activity = torch.tensor([3, 0, 5, 0, 1, 3])
            >>> encoder_weight = Parameter(torch.ones((6, 2)))
            >>> rescaled_input = ActivationResampler.renormalize_and_scale(
            ...     sampled_input,
            ...     neuron_activity,
            ...     encoder_weight
            ... )
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

        # Handle no dead neurons
        n_dead_neurons = len(sampled_input)
        if n_dead_neurons == 0:
            return torch.empty(
                (0, sampled_input.shape[-1]), dtype=sampled_input.dtype, device=sampled_input.device
            )

        # Calculate the average norm of the encoder weights for alive neurons.
        detached_encoder_weight = encoder_weight.detach()  # Don't track gradients
        alive_encoder_weights: Float[
            Tensor, Axis.names(Axis.ALIVE_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
        ] = detached_encoder_weight[alive_neuron_mask, :]
        average_alive_norm: Float[Tensor, Axis.SINGLE_ITEM] = alive_encoder_weights.norm(
            dim=-1
        ).mean()

        # Renormalize the input vector to equal the average norm of the encoder weights for alive
        # neurons times 0.2.
        renormalized_input: Float[
            Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
        ] = torch.nn.functional.normalize(sampled_input, dim=-1)
        return renormalized_input * (average_alive_norm * 0.2)

    def resample_dead_neurons(
        self,
        activation_store: ActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> list[ParameterUpdateResults]:
        """Resample dead neurons.

        Args:
            activation_store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).

        Returns:
            For each component that the SAE is being trained on, the indices of dead neurons and the
            updates for the encoder and decoder weights and biases.
        """
        parameter_update_results: list[ParameterUpdateResults] = []

        with torch.no_grad():
            dead_neuron_indices: list[
                Int64[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)]
            ] = self._get_dead_neuron_indices()

            # Compute the loss for the current model on a random subset of inputs and get the
            # activations.
            loss, input_activations = self.compute_loss_and_get_activations(
                store=activation_store,
                autoencoder=autoencoder,
                loss_fn=loss_fn,
                train_batch_size=train_batch_size,
            )

            # Assign each input vector a probability of being picked that is proportional to the
            # square of the autoencoder's loss on that input.
            sample_probabilities: Float[
                Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)
            ] = self.assign_sampling_probabilities(loss)

            # For each dead neuron sample an input according to these probabilities.
            sampled_input: list[
                Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]
            ] = self.sample_input(
                sample_probabilities, input_activations, [len(dead) for dead in dead_neuron_indices]
            )

            for component_idx in range(self._n_components):
                # Renormalize each input vector to have unit L2 norm and set this to be the
                # dictionary vector for the dead autoencoder neuron.
                renormalized_input: Float[
                    Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
                ] = torch.nn.functional.normalize(sampled_input[component_idx], dim=-1)

                dead_decoder_weight_updates = rearrange(
                    renormalized_input, "dead_neuron input_feature -> input_feature dead_neuron"
                )

                # For the corresponding encoder vector, renormalize the input vector to equal the
                # average norm of the encoder weights for alive neurons times 0.2. Set the
                # corresponding encoder bias element to zero.
                encoder_weight: Float[
                    Tensor, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
                ] = get_component_slice_tensor(autoencoder.encoder.weight, 3, 0, component_idx)

                rescaled_sampled_input = self.renormalize_and_scale(
                    sampled_input=sampled_input[component_idx],
                    neuron_activity=self._collated_neuron_activity[component_idx],
                    encoder_weight=encoder_weight,
                )

                dead_encoder_bias_updates = torch.zeros_like(
                    dead_neuron_indices[component_idx],
                    dtype=dead_decoder_weight_updates.dtype,
                    device=dead_decoder_weight_updates.device,
                )

                parameter_update_results.append(
                    ParameterUpdateResults(
                        dead_neuron_indices=dead_neuron_indices[component_idx],
                        dead_encoder_weight_updates=rescaled_sampled_input,
                        dead_encoder_bias_updates=dead_encoder_bias_updates,
                        dead_decoder_weight_updates=dead_decoder_weight_updates,
                    )
                )

            return parameter_update_results

    def step_resampler(
        self,
        batch_neuron_activity: Int64[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        activation_store: ActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> list[ParameterUpdateResults] | None:
        """Step the resampler, collating neuron activity and resampling if necessary.

        Args:
            batch_neuron_activity: Number of times each neuron fired in the current batch.
            activation_store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).

        Returns:
            Parameter update results (for each component that the SAE is being trained on) if
            resampling is due. Otherwise None.
        """
        # Update the counter
        self._activations_seen_since_last_resample += len(activation_store)

        if self._n_times_resampled < self._max_n_resamples:
            # Collate neuron activity, if in the data collection window. For example in the
            # Anthropic Towards Monosemanticity paper, the window started collecting at 100m
            # activations and stopped at 200m (and then repeated this again a few times until the
            # max times to resample was hit).
            if self._activations_seen_since_last_resample >= self.neuron_activity_window_start:
                detached_neuron_activity = batch_neuron_activity.detach().cpu()
                self._collated_neuron_activity.add_(detached_neuron_activity)
                self._n_activations_collated_since_last_resample += train_batch_size

            # Check if we should resample.
            if self._activations_seen_since_last_resample >= self.neuron_activity_window_end:
                # Get resampled dictionary vectors
                resample_res = self.resample_dead_neurons(
                    activation_store=activation_store,
                    autoencoder=autoencoder,
                    loss_fn=loss_fn,
                    train_batch_size=train_batch_size,
                )

                # Update counters
                self._activations_seen_since_last_resample = 0
                self._n_activations_collated_since_last_resample = 0
                self._n_times_resampled += 1

                # Reset the collated neuron activity
                self._collated_neuron_activity.zero_()

                return resample_res

        return None

    def __str__(self) -> str:
        """Return a string representation of the activation resampler."""
        return (
            f"ActivationResampler("
            f"n_components={self._n_components}, "
            f"neuron_activity_window_start={self.neuron_activity_window_end}, "
            f"neuron_activity_window_end={self.neuron_activity_window_end}, "
            f"max_resamples={self._max_n_resamples}, "
            f"resample_dataset_size={self._resample_dataset_size}, "
            f"dead_neuron_threshold={self._threshold_is_dead_portion_fires})"
        )
