"""Abstract train metric."""
from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis


class AbstractTrainMetric(Metric, ABC):
    """Abstract train metric.

    Extends TorchMetrics Metric class, which is stateful so that it works in distributed
    environments.

    https://lightning.ai/docs/torchmetrics/stable/
    """

    _component_names: list[str]
    _metric_name: str

    def __init__(
        self,
        component_names: list[str],
        metric_name: str,
        *,
        compute_on_cpu: bool = False,
        compute_with_cache: bool = True,
        dist_sync_on_step: bool = False,
        sync_on_compute: bool = True,
    ) -> None:
        """Initialize the metric.

        https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metric-kwargs

        Args:
            component_names: Names of the components that the SAE is trained on.
            metric_name: Name of the metric.
            compute_on_cpu: Will automatically move the metric states to cpu after calling update,
                making sure that GPU memory is not filling up. The consequence will be that the
                compute method will be called on CPU instead of GPU. Only applies to metric states
                that are lists.
            compute_with_cache: This argument indicates if the result after calling the compute
                method should be cached. By default this is True meaning that repeated calls to
                compute (with no change to the metric state in between) does not recompute the
                metric but just returns the cache. By setting it to False the metric will be
                recomputed every time compute is called, but it can also help clean up a bit of
                memory.
            dist_sync_on_step: This argument is bool that indicates if the metric should
                synchronize between different devices every time forward is called. Setting this to
                True is in general not recommended as synchronization is an expensive operation to
                do after each batch.
            sync_on_compute: This argument is an bool that indicates if the metrics should
                automatically sync between devices whenever the compute method is called. By default
                this is True, but by setting this to False you can manually control when the
                synchronization happens.
        """
        super().__init__(
            compute_on_cpu=compute_on_cpu,
            compute_with_cache=compute_with_cache,
            dist_sync_on_step=dist_sync_on_step,
            sync_on_compute=sync_on_compute,
        )

        self._component_names = component_names
        self._metric_name = metric_name

    @abstractmethod
    def update(
        self,
        input_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> None:
        """Update the metric state.

        Args:
            input_activations: The input activations.
            learned_activations: The learned activations.
            decoded_activations: The decoded activations.
        """
