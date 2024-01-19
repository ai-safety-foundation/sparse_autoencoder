"""Train batch feature density."""
import einops
from jaxtyping import Float
from pydantic import NonNegativeFloat, validate_call
import torch
from torch import Tensor

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
)
from sparse_autoencoder.tensor_types import Axis


class TrainBatchFeatureDensityMetric(AbstractTrainMetric):
    """Train batch feature density.

    Percentage of samples in which each feature was active (i.e. the neuron has "fired"), in a
    training batch.

    Generally we want a small number of features to be active in each batch, so average feature
    density should be low. By contrast if the average feature density is high, it means that the
    features are not sparse enough.

    Warning:
        This is not the same as the feature density of the entire training set. It's main use is
        tracking the progress of training.
    """

    _component_names: list[str]
    full_state_update: bool | None = False
    process_results: list[Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]]
    threshold: float

    @validate_call
    def __init__(
        self,
        component_names: list[str],
        threshold: NonNegativeFloat = 0.0,
    ) -> None:
        """Initialise the train batch feature density metric.

        Args:
            component_names: Names of the components.
            threshold: Threshold for considering a feature active (i.e. the neuron has "fired").
                This should be close to zero.
        """
        super().__init__(component_names, "feature_density")
        self._component_names = component_names
        self.threshold = threshold

    def feature_density(
        self,
        learned_activations: Float[
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]:
        """Count how many times each feature was active.

        Percentage of samples in which each feature was active (i.e. the neuron has "fired").

        Example:
            >>> import torch
            >>> activations = torch.tensor([[[0.5, 0.5, 0.0]], [[0.5, 0.0, 0.0001]]])
            >>> TrainBatchFeatureDensityMetric(["mlp_1"], 0.001).feature_density(activations).tolist()
            [[1.0, 0.5, 0.0]]

        Args:
            learned_activations: Sample of cached activations (the Autoencoder's learned features).

        Returns:
            Number of times each feature was active in a sample.
        """
        has_fired: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.gt(learned_activations, self.threshold).to(
            dtype=torch.float  # Move to float so it can be averaged
        )

        return einops.reduce(
            has_fired,
            f"{Axis.BATCH} {Axis.COMPONENT} {Axis.LEARNT_FEATURE} \
                -> {Axis.COMPONENT} {Axis.LEARNT_FEATURE}",
            "mean",
        )

    def update(
        self,
        input_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> None:
        """Update the metric state.

        Args:
            input_activations: The input activations.
            learned_activations: The learned activations.
            decoded_activations: The decoded activations.
        """
        process_res = self.feature_density(learned_activations)
        self.process_results.append(process_res)

    def compute(
        self,
    ) -> dict[str, float | Tensor]:
        """Compute the metric."""
        all_process_results = torch.stack(self.process_results)
        process_average: Float[Tensor, Axis.COMPONENT] = einops.reduce(
            all_process_results,
            (
                f"{Axis.PROCESS} {Axis.COMPONENT} {Axis.LEARNT_FEATURE} "
                f"-> {Axis.COMPONENT} {Axis.LEARNT_FEATURE}"
            ),
            "mean",
        )

        results: dict[str, Float] = dict(zip(self._component_names, process_average))
        results["total"] = process_average.mean(0)
        return results
