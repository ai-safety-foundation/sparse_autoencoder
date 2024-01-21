"""PyTorch Lightning module for training a sparse autoencoder."""
from functools import partial

from jaxtyping import Float
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchmetrics import ClasswiseWrapper, MetricCollection

from sparse_autoencoder.autoencoder.model import (
    ForwardPassResult,
    SparseAutoencoder,
    SparseAutoencoderConfig,
)
from sparse_autoencoder.metrics.loss.l1_absolute_loss import L1AbsoluteLoss
from sparse_autoencoder.metrics.loss.l2_reconstruction_loss import L2ReconstructionLoss
from sparse_autoencoder.metrics.train.feature_density import FeatureDensityMetric
from sparse_autoencoder.metrics.train.l0_norm import L0NormMetric
from sparse_autoencoder.metrics.train.neuron_activity import NeuronActivityMetric
from sparse_autoencoder.metrics.train.neuron_fired_count import NeuronFiredCountMetric
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset
from sparse_autoencoder.tensor_types import Axis


class LitSparseAutoencoder(LightningModule):
    """Lightning Sparse Autoencoder."""

    neuron_fired_count: NeuronFiredCountMetric

    def __init__(
        self,
        config: SparseAutoencoderConfig,
        component_names: list[str],
        l1_coefficient: float = 0.001,
    ):
        """Initialise the module."""
        super().__init__()
        self.sparse_autoencoder = SparseAutoencoder(config)
        self._l1_coefficient = l1_coefficient

        num_components = config.n_components or 1
        add_component_names = partial(ClasswiseWrapper, labels=component_names, prefix="train/")

        # Create the loss metrics
        self.l1_loss = L1AbsoluteLoss(num_components=num_components)
        self.l2_loss = L2ReconstructionLoss(num_components=num_components)

        self.neuron_fired_count = NeuronFiredCountMetric(
            num_learned_features=config.n_learned_features,
            num_components=num_components,
        )

        self.train_metrics = MetricCollection(
            [
                add_component_names(L0NormMetric(num_components=num_components)),
                add_component_names(
                    NeuronActivityMetric(
                        num_learned_features=config.n_learned_features,
                        num_components=num_components,
                    ),
                ),
                add_component_names(
                    FeatureDensityMetric(
                        num_learned_features=config.n_learned_features,
                        num_components=num_components,
                    ),
                ),
            ]
        )

    def forward(  # type: ignore[override]
        self,
        inputs: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> ForwardPassResult:
        """Forward pass."""
        return self.model.forward(inputs)

    def training_step(  # type: ignore[override]
        self,
        batch: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> None:
        """Training step."""
        # Forward pass
        output: ForwardPassResult = self(batch)

        # Metrics
        train_metrics = self.train_metrics(learned_activations=output.learned_activations)

        # Loss
        l1_loss = self.l1_loss(learned_activations=output.learned_activations)
        l2_loss = self.l2_loss(
            source_activations=batch, decoded_activations=output.decoded_activations
        )
        component_wise_loss = l1_loss.values() * self._l1_coefficient + l2_loss.values()
        loss = train_metrics["loss"]

        # Log
        component_wise_loss_metrics = {
            f"train/loss/{name}": value
            for name, value in zip(self.component_names, component_wise_loss)
        }
        self.log_dict({**train_metrics, **l1_loss, **l2_loss, **component_wise_loss_metrics})

        # Neuron activity
        self.neuron_fired_count.update(learned_activations=output.learned_activations)

        return loss

    def on_after_backward(self) -> None:
        """After-backward pass hook."""
        self.autoencoder.post_backwards_hook()

    def on_train_start(self) -> None:
        """Train start hook."""
        # Reset the neuron fired count
        self.neuron_fired_count.reset()

    def configure_optimizers(self) -> Optimizer:
        """Configure the optimizer."""
        return AdamWithReset(
            self.parameters(), named_parameters=self.named_parameters(), has_components_dim=True
        )
