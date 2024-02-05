"""PyTorch Lightning module for training a sparse autoencoder."""
from functools import partial
from typing import Any

from jaxtyping import Float
from lightning.pytorch import LightningModule
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
import wandb

from sparse_autoencoder.activation_resampler.activation_resampler import (
    ActivationResampler,
    ParameterUpdateResults,
)
from sparse_autoencoder.autoencoder.model import (
    ForwardPassResult,
    SparseAutoencoder,
    SparseAutoencoderConfig,
)
from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.metrics.loss.l1_absolute_loss import L1AbsoluteLoss
from sparse_autoencoder.metrics.loss.l2_reconstruction_loss import L2ReconstructionLoss
from sparse_autoencoder.metrics.loss.sae_loss import SparseAutoencoderLoss
from sparse_autoencoder.metrics.train.l0_norm import L0NormMetric
from sparse_autoencoder.metrics.train.neuron_activity import NeuronActivityMetric
from sparse_autoencoder.metrics.wrappers.classwise import ClasswiseWrapperWithMean
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset
from sparse_autoencoder.tensor_types import Axis


class LitSparseAutoencoderConfig(SparseAutoencoderConfig):
    """PyTorch Lightning Sparse Autoencoder config."""

    component_names: list[str]

    l1_coefficient: float = 0.001

    resample_interval: PositiveInt = 200000000

    max_n_resamples: NonNegativeInt = 4

    resample_dead_neurons_dataset_size: PositiveInt = 100000000

    resample_loss_dataset_size: PositiveInt = 819200

    resample_threshold_is_dead_portion_fires: NonNegativeFloat = 0.0

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Model post init validation.

        Args:
            __context: Pydantic context.

        Raises:
            ValueError: If the number of component names does not match the number of components.
        """
        if self.n_components and len(self.component_names) != self.n_components:
            error_message = (
                f"Number of component names ({len(self.component_names)}) must match the number of "
                f"components ({self.n_components})"
            )
            raise ValueError(error_message)


class LitSparseAutoencoder(LightningModule):
    """Lightning Sparse Autoencoder."""

    sparse_autoencoder: SparseAutoencoder

    config: LitSparseAutoencoderConfig

    loss_fn: SparseAutoencoderLoss

    train_metrics: MetricCollection

    def __init__(
        self,
        config: LitSparseAutoencoderConfig,
    ):
        """Initialise the module."""
        super().__init__()
        self.sparse_autoencoder = SparseAutoencoder(config)
        self.config = config

        num_components = config.n_components or 1
        add_component_names = partial(
            ClasswiseWrapperWithMean, component_names=config.component_names
        )

        # Create the loss & metrics
        self.loss_fn = SparseAutoencoderLoss(
            num_components, config.l1_coefficient, keep_batch_dim=True
        )

        self.train_metrics = MetricCollection(
            {
                "l0": add_component_names(L0NormMetric(num_components), prefix="train/l0_norm"),
                "activity": add_component_names(
                    NeuronActivityMetric(config.n_learned_features, num_components),
                    prefix="train/neuron_activity",
                ),
                "l1": add_component_names(
                    L1AbsoluteLoss(num_components), prefix="loss/l1_learned_activations"
                ),
                "l2": add_component_names(
                    L2ReconstructionLoss(num_components), prefix="loss/l2_reconstruction"
                ),
                "loss": add_component_names(
                    SparseAutoencoderLoss(num_components, config.l1_coefficient),
                    prefix="loss/total",
                ),
            },
            # Share state & updates across groups (to avoid e.g. computing l1 twice for both the
            # loss and l1 metrics). Note the metric that goes first must calculate all the states
            # needed by the rest of the group.
            compute_groups=[
                ["loss", "l1", "l2"],
                ["activity"],
                ["l0"],
            ],
        )

        self.activation_resampler = ActivationResampler(
            n_learned_features=config.n_learned_features,
            n_components=num_components,
            resample_interval=config.resample_interval,
            max_n_resamples=config.max_n_resamples,
            n_activations_activity_collate=config.resample_dead_neurons_dataset_size,
            resample_dataset_size=config.resample_loss_dataset_size,
            threshold_is_dead_portion_fires=config.resample_threshold_is_dead_portion_fires,
        )

    def forward(  # type: ignore[override]
        self,
        inputs: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> ForwardPassResult:
        """Forward pass."""
        return self.sparse_autoencoder.forward(inputs)

    def update_parameters(self, parameter_updates: list[ParameterUpdateResults]) -> None:
        """Update the parameters of the model from the results of the resampler.

        Args:
            parameter_updates: Parameter updates from the resampler.

        Raises:
            TypeError: If the optimizer is not an AdamWithReset.
        """
        for component_idx, component_parameter_update in enumerate(parameter_updates):
            # Update the weights and biases
            self.sparse_autoencoder.encoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_weight_updates,
                component_idx=component_idx,
            )
            self.sparse_autoencoder.encoder.update_bias(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_bias_updates,
                component_idx=component_idx,
            )
            self.sparse_autoencoder.decoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_decoder_weight_updates,
                component_idx=component_idx,
            )

            # Reset the optimizer
            for (
                parameter,
                axis,
            ) in self.reset_optimizer_parameter_details:
                optimizer = self.optimizers(use_pl_optimizer=False)
                if not isinstance(optimizer, AdamWithReset):
                    error_message = "Cannot reset the optimizer. "
                    raise TypeError(error_message)

                optimizer.reset_neurons_state(
                    parameter=parameter,
                    neuron_indices=component_parameter_update.dead_neuron_indices,
                    axis=axis,
                    component_idx=component_idx,
                )

    def training_step(  # type: ignore[override]
        self,
        batch: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        batch_idx: int | None = None,  # noqa: ARG002
    ) -> Float[Tensor, Axis.SINGLE_ITEM]:
        """Training step."""
        # Forward pass
        output: ForwardPassResult = self.forward(batch)

        # Metrics & loss
        train_metrics = self.train_metrics.forward(
            source_activations=batch,
            learned_activations=output.learned_activations,
            decoded_activations=output.decoded_activations,
        )

        loss = self.loss_fn.forward(
            source_activations=batch,
            learned_activations=output.learned_activations,
            decoded_activations=output.decoded_activations,
        )

        if wandb.run is not None:
            self.log_dict(train_metrics)

        # Resample dead neurons
        parameter_updates = self.activation_resampler.forward(
            input_activations=batch,
            learned_activations=output.learned_activations,
            loss=loss,
            encoder_weight_reference=self.sparse_autoencoder.encoder.weight,
        )
        if parameter_updates is not None:
            self.update_parameters(parameter_updates)

        # Return the mean loss
        return loss.mean()

    def on_after_backward(self) -> None:
        """After-backward pass hook."""
        self.sparse_autoencoder.post_backwards_hook()

    def configure_optimizers(self) -> Optimizer:
        """Configure the optimizer."""
        return AdamWithReset(
            self.sparse_autoencoder.parameters(),
            named_parameters=self.sparse_autoencoder.named_parameters(),
            has_components_dim=True,
        )

    @property
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
        """Reset optimizer parameter details."""
        return self.sparse_autoencoder.reset_optimizer_parameter_details
