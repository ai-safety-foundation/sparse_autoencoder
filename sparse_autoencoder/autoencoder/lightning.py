"""PyTorch Lightning module for training a sparse autoencoder."""
from functools import partial

from jaxtyping import Float
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

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
from sparse_autoencoder.metrics.wrappers.classwise import ClasswiseWrapperWithMean
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset
from sparse_autoencoder.tensor_types import Axis


class LitSparseAutoencoder(LightningModule):
    """Lightning Sparse Autoencoder."""

    neuron_fired_count: NeuronFiredCountMetric

    _keep_batch_dim_loss: bool = False

    @property
    def keep_batch_dim_loss(self) -> bool:
        """Whether to keep the batch dimension in the loss output."""
        return self._keep_batch_dim_loss

    @keep_batch_dim_loss.setter
    def keep_batch_dim_loss(self, keep_batch_dim_loss: bool) -> None:
        """Set whether to keep the batch dimension in the loss output."""
        self.l1_loss.keep_batch_dim = keep_batch_dim_loss
        self.l2_loss.keep_batch_dim = keep_batch_dim_loss

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
        add_component_names = partial(ClasswiseWrapperWithMean, labels=component_names)

        # Create the loss & metrics
        self.l1_loss = L1AbsoluteLoss(num_components)
        self.l2_loss = L2ReconstructionLoss(num_components)
        loss = l1_coefficient * self.l1_loss + self.l2_loss

        self.neuron_fired_count = NeuronFiredCountMetric(
            num_learned_features=config.n_learned_features, num_components=num_components
        )

        self.train_metrics = MetricCollection(
            {
                "l0": add_component_names(L0NormMetric(num_components), prefix="train/l0_norm/"),
                "activity": add_component_names(
                    NeuronActivityMetric(config.n_learned_features, num_components),
                    prefix="train/neuron_activity/",
                ),
                "density": add_component_names(
                    FeatureDensityMetric(config.n_learned_features, num_components),
                    prefix="train/feature_density/",
                ),
                "l1": add_component_names(self.l1_loss, prefix="loss/l1_learned_activations/"),
                "l2": add_component_names(self.l2_loss, prefix="loss/l2_reconstruction/"),
                "loss": add_component_names(loss, prefix="loss/total/"),
            }
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
        batch_idx: int | None = None,  # noqa: ARG002
    ) -> (
        Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)]
        | Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]
    ):
        """Training step."""
        # Forward pass
        output: ForwardPassResult = self(batch)

        # Metrics & loss
        train_metrics = self.train_metrics(
            source_activations=batch,
            learned_activations=output.learned_activations,
            decoded_activations=output.decoded_activations,
        )
        self.log_dict(train_metrics)

        # Neuron activity
        self.neuron_fired_count.update(learned_activations=output.learned_activations)

        return train_metrics["loss/total/mean"]

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
