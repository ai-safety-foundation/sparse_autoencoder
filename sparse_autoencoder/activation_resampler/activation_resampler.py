"""Activation resampler."""
from dataclasses import dataclass
from typing import Annotated

from einops import rearrange
from jaxtyping import Bool, Float, Int
from pydantic import Field, NonNegativeInt, PositiveInt, validate_call
import torch
from torch import Tensor, distributed
from torch.distributed import get_world_size, group
from torch.nn import Parameter
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from sparse_autoencoder.activation_resampler.utils.component_slice_tensor import (
    get_component_slice_tensor,
)
from sparse_autoencoder.tensor_types import Axis


@dataclass
class ParameterUpdateResults:
    """Parameter update results from resampling dead neurons."""

    dead_neuron_indices: Int[Tensor, Axis.LEARNT_FEATURE_IDX]
    """Dead neuron indices."""

    dead_encoder_weight_updates: Float[
        Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Dead encoder weight updates."""

    dead_encoder_bias_updates: Float[Tensor, Axis.DEAD_FEATURE]
    """Dead encoder bias updates."""

    dead_decoder_weight_updates: Float[
        Tensor, Axis.names(Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)
    ]
    """Dead decoder weight updates."""


class ActivationResampler(Metric):
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

    # Collated data from the train loop
    _neuron_fired_count: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    _loss: list[Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL)]] | Float[
        Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL)
    ]
    _input_activations: list[
        Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]
    ] | Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]

    # Tracking
    _n_activations_seen_process: int
    _n_times_resampled: int

    # Settings
    _n_components: int
    _threshold_is_dead_portion_fires: float
    _max_n_resamples: int
    resample_interval: int
    resample_interval_process: int
    start_collecting_neuron_activity_process: int
    start_collecting_loss_process: int

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
        super().__init__(
            sync_on_compute=False  # Manually sync instead in compute, where needed
        )

        # Error handling
        if n_activations_activity_collate > resample_interval:
            error_message = "Must collate less activation activity than the resample interval."
            raise ValueError(error_message)

        # Number of processes
        world_size = (
            get_world_size(group.WORLD)
            if distributed.is_available() and distributed.is_initialized()
            else 1
        )
        process_resample_dataset_size = resample_dataset_size // world_size

        # State setup (note half precision is used as it's sufficient for resampling purposes)
        self.add_state(
            "_neuron_fired_count",
            torch.zeros((n_components, n_learned_features)),
            "sum",
        )
        self.add_state("_loss", [], "cat")
        self.add_state("_input_activations", [], "cat")

        # Tracking
        self._n_activations_seen_process = 0
        self._n_times_resampled = 0

        # Settings
        self._n_components = n_components
        self._threshold_is_dead_portion_fires = threshold_is_dead_portion_fires
        self._max_n_resamples = max_n_resamples
        self.resample_interval = resample_interval
        self.resample_interval_process = resample_interval // world_size
        self.start_collecting_neuron_activity_process = (
            self.resample_interval_process - n_activations_activity_collate // world_size
        )
        self.start_collecting_loss_process = (
            self.resample_interval_process - process_resample_dataset_size
        )

    def update(
        self,
        input_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        loss: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)],
        encoder_weight_reference: Parameter,
    ) -> None:
        """Update the collated data from forward passes.

        Args:
            input_activations: Input activations to the SAE.
            learned_activations: Learned activations from the SAE.
            loss: Loss per input activation.
            encoder_weight_reference: Reference to the SAE encoder weight tensor.

        Raises:
            TypeError: If the loss or input activations are not lists (e.g. from unsync having not
                been called).
        """
        if self._n_activations_seen_process >= self.start_collecting_neuron_activity_process:
            neuron_has_fired: Bool[
                Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
            ] = torch.gt(learned_activations, 0)
            self._neuron_fired_count += neuron_has_fired.sum(dim=0)

        if self._n_activations_seen_process >= self.start_collecting_loss_process:
            # Typecast
            if not isinstance(self._loss, list) or not isinstance(self._input_activations, list):
                raise TypeError

            # Append
            self._loss.append(loss)
            self._input_activations.append(input_activations)

        self._n_activations_seen_process += len(learned_activations)
        self._encoder_weight = encoder_weight_reference

    def _get_dead_neuron_indices(
        self,
    ) -> list[Int[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)]]:
        """Identify the indices of neurons that are dead.

        Identifies any neurons that have fired less than the threshold portion of the collated
        sample size.

        Returns:
            List of dead neuron indices for each component.

        Raises:
            ValueError: If no neuron activity has been collated yet.
        """
        # Check we have already collated some neuron activity
        if torch.all(self._neuron_fired_count == 0):
            error_message = "Cannot get dead neuron indices without neuron activity."
            raise ValueError(error_message)

        # Find any neurons that fire less than the threshold portion of times
        threshold_is_dead_n_fires: int = int(
            self.resample_interval * self._threshold_is_dead_portion_fires
        )

        return [
            torch.where(self._neuron_fired_count[component_idx] <= threshold_is_dead_n_fires)[0].to(
                dtype=torch.int
            )
            for component_idx in range(self._n_components)
        ]

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
            component_sample_indices: Int[Tensor, Axis.LEARNT_FEATURE_IDX] = torch.multinomial(
                component_probabilities, num_samples=component_n_samples
            )
            sampled_inputs.append(component_input_activations[component_sample_indices, :])

        return sampled_inputs

    @staticmethod
    def renormalize_and_scale(
        sampled_input: Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)],
        neuron_activity: Float[Tensor, Axis.names(Axis.LEARNT_FEATURE)],
        encoder_weight: Float[Tensor, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)],
    ) -> Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]:
        """Renormalize and scale the resampled dictionary vectors.

        Renormalize the input vector to equal the average norm of the encoder weights for alive
        neurons times 0.2.

        Example:
            >>> from torch.nn import Parameter
            >>> _seed = torch.manual_seed(0)  # For reproducibility in example
            >>> sampled_input = torch.tensor([[3.0, 4.0]])
            >>> neuron_activity = torch.tensor([3.0, 0, 5, 0, 1, 3])
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

    def compute(self) -> list[ParameterUpdateResults] | None:
        """Compute the parameters that need to be updated.

        Returns:
            A list of parameter update results (for each component that the SAE is being trained
            on), if an update is needed.
        """
        # Resample if needed
        if self._n_activations_seen_process >= self.resample_interval_process:
            with torch.no_grad():
                # Initialise results
                parameter_update_results: list[ParameterUpdateResults] = []

                # Sync & typecast
                self.sync()
                loss = dim_zero_cat(self._loss)
                input_activations = dim_zero_cat(self._input_activations)

                dead_neuron_indices: list[
                    Int[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)]
                ] = self._get_dead_neuron_indices()

                # Assign each input vector a probability of being picked that is proportional to the
                # square of the autoencoder's loss on that input.
                sample_probabilities: Float[
                    Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)
                ] = self.assign_sampling_probabilities(loss)

                # For each dead neuron sample an input according to these probabilities.
                sampled_input: list[
                    Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]
                ] = self.sample_input(
                    sample_probabilities,
                    input_activations,
                    [len(dead) for dead in dead_neuron_indices],
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

                    # For the corresponding encoder vector, renormalize the input vector to equal
                    # the average norm of the encoder weights for alive neurons times 0.2. Set the
                    # corresponding encoder bias element to zero.
                    encoder_weight: Float[
                        Tensor, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
                    ] = get_component_slice_tensor(self._encoder_weight, 3, 0, component_idx)

                    rescaled_sampled_input = self.renormalize_and_scale(
                        sampled_input=sampled_input[component_idx],
                        neuron_activity=self._neuron_fired_count[component_idx],
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

                # Reset
                self.unsync(should_unsync=self._is_synced)
                self.reset()

                return parameter_update_results

        return None

    def forward(  # type: ignore[override]
        self,
        input_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        loss: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)],
        encoder_weight_reference: Parameter,
    ) -> list[ParameterUpdateResults] | None:
        """Step the resampler, collating neuron activity and resampling if necessary.

        Args:
            input_activations: Input activations to the SAE.
            learned_activations: Learned activations from the SAE.
            loss: Loss per input activation.
            encoder_weight_reference: Reference to the SAE encoder weight tensor.

        Returns:
            Parameter update results (for each component that the SAE is being trained on) if
            resampling is due. Otherwise None.
        """
        # Don't do anything if we have already completed all resamples
        if self._n_times_resampled >= self._max_n_resamples:
            return None

        self.update(
            input_activations=input_activations,
            learned_activations=learned_activations,
            loss=loss,
            encoder_weight_reference=encoder_weight_reference,
        )

        return self.compute()

    def reset(self) -> None:
        """Reset the activation resampler.

        Warning:
            This is only called when forward/compute has returned parameters to update (i.e.
            resampling is due).
        """
        self._n_activations_seen_process = 0
        self._neuron_fired_count = torch.zeros_like(self._neuron_fired_count)
        self._loss = []
        self._input_activations = []
        self._n_times_resampled += 1
