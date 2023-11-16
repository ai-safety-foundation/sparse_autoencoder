"""Abstract pipeline."""
from abc import ABC, abstractmethod
from typing import final

from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
)
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.metrics.abstract_metric import (
    AbstractGenerateMetric,
    AbstractTrainMetric,
    AbstractValidationMetric,
)
from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset
from sparse_autoencoder.tensor_types import NeuronActivity


class AbstractPipeline(ABC):
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    generate_metrics: list[AbstractGenerateMetric]

    train_metrics: list[AbstractTrainMetric]

    validation_metric: list[AbstractValidationMetric]

    source_model: HookedTransformer

    source_dataset: SourceDataset

    autoencoder: SparseAutoencoder

    loss: AbstractLoss

    optimizer: AbstractOptimizerWithReset

    activation_resampler: AbstractActivationResampler | None

    progress_bar: tqdm | None

    @final
    def __init__(
        self,
        generate_metrics: list[AbstractGenerateMetric],
        train_metrics: list[AbstractTrainMetric],
        validation_metric: list[AbstractValidationMetric],
        source_model: HookedTransformer,
        autoencoder: SparseAutoencoder,
        source_dataset: SourceDataset,
        activation_resampler: AbstractActivationResampler | None,
        optimizer: AbstractOptimizerWithReset,
        loss: AbstractLoss,
    ):
        """Initialize the pipeline."""
        self.generate_metrics = generate_metrics
        self.train_metrics = train_metrics
        self.validation_metric = validation_metric
        self.source_model = source_model
        self.autoencoder = autoencoder
        self.source_dataset = source_dataset
        self.activation_resampler = activation_resampler
        self.optimizer = optimizer
        self.loss = loss

    @abstractmethod
    def generate_activations(self) -> TensorActivationStore:
        """Generate activations."""
        raise NotImplementedError

    @abstractmethod
    def train_autoencoder(self, activations: TensorActivationStore) -> NeuronActivity:
        """Train the sparse autoencoder."""
        raise NotImplementedError

    @abstractmethod
    def resample_neurons(self, neuron_activity: NeuronActivity) -> None:
        """Resample dead neurons."""
        raise NotImplementedError

    def validate_sae(self) -> None:
        """Get validation metrics."""
        raise NotImplementedError

    @final
    def run_pipeline(
        self,
        source_batch_size: int,
        resample_frequency: int,
        validate_frequency: int,
        checkpoint_frequency: int,
        max_activations: int,
    ) -> None:
        """Run the full training pipeline."""
        last_resampled: int = 0
        last_validated: int = 0
        last_checkpoint: int = 0
        neuron_activity: NeuronActivity | None = None

        for _ in tqdm(range(0, max_activations, source_batch_size), title="Activations trained on"):
            # Generate
            activations: TensorActivationStore = self.generate_activations()

            # Train
            batch_neuron_activity: NeuronActivity = self.train_autoencoder(activations)
            detached_neuron_activity = batch_neuron_activity.detach().cpu()
            if neuron_activity:
                neuron_activity.add_(detached_neuron_activity)
            else:
                neuron_activity = detached_neuron_activity

            # Resample dead neurons (if needed)
            if last_resampled > resample_frequency:
                self.resample_neurons(neuron_activity)
                self.last_resampled = 0

            # Get validation metrics (if needed)
            if last_validated > validate_frequency:
                self.validate_sae()
                self.last_validated = 0

            # Checkpoint (if needed)
            if last_checkpoint > checkpoint_frequency:
                self.autoencoder.save_to_hf()
                self.last_checkpoint = 0
