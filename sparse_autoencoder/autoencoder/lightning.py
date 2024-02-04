"""PyTorch Lightning module for training a sparse autoencoder."""
from functools import partial

from jaxtyping import Float
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
import wandb

from sparse_autoencoder.activation_resampler.activation_resampler import ActivationResampler
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


class LitSparseAutoencoder(LightningModule):
    """Lightning Sparse Autoencoder."""

    sparse_autoencoder: SparseAutoencoder

    config: SparseAutoencoderConfig

    loss_fn: SparseAutoencoderLoss

    train_metrics: MetricCollection

    def __init__(
        self,
        config: SparseAutoencoderConfig,
        component_names: list[str],
        l1_coefficient: float = 0.001,
    ):
        """Initialise the module."""
        super().__init__()
        self.sparse_autoencoder = SparseAutoencoder(config)
        self.config = self.sparse_autoencoder.config

        num_components = config.n_components or 1
        add_component_names = partial(ClasswiseWrapperWithMean, component_names=component_names)

        # Create the loss & metrics
        self.loss_fn = SparseAutoencoderLoss(num_components, l1_coefficient, keep_batch_dim=True)

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
                    SparseAutoencoderLoss(num_components, l1_coefficient), prefix="loss/total"
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
            encoder_weight_reference=self.model.encoder.weight,
            n_components=num_components,
        )

    def forward(  # type: ignore[override]
        self,
        inputs: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> ForwardPassResult:
        """Forward pass."""
        return self.sparse_autoencoder.forward(inputs)

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

        # Resampler
        parameter_updates = self.activation_resampler.forward(
            input_activations=batch,
            learned_activations=output.learned_activations,
            loss=loss,
        )
        if parameter_updates is not None:
            self.model.update_parameters(parameter_updates)

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
        return self.model.reset_optimizer_parameter_details
