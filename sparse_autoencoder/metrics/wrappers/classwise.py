"""Classwise metrics wrapper."""
import torch
from torch import Tensor
from torchmetrics import ClasswiseWrapper, Metric


class ClasswiseWrapperWithMean(ClasswiseWrapper):
    """Classwise wrapper with mean.

    This metric works together with classification metrics that returns multiple values (one value
    per class) such that label information can be automatically included in the output. It extends
    the standard torchmetrics wrapper that does this, adding in an additional mean value (across all
    classes).
    """

    _prefix: str

    labels: list[str]

    def __init__(
        self,
        metric: Metric,
        component_names: list[str] | None = None,
        prefix: str | None = None,
    ) -> None:
        """Initialise the classwise wrapper.

        Args:
            metric: Metric to wrap.
            component_names: Component names.
            prefix: Prefix for the name (will replace the default of the class name).
        """
        super().__init__(metric, component_names, prefix)

        # Default prefix
        if not self._prefix:
            self._prefix = f"{self.metric.__class__.__name__.lower()}"

    def _convert(self, x: Tensor) -> dict[str, Tensor]:
        """Convert the input tensor to a dictionary of metrics.

        Args:
            x: The input tensor.

        Returns:
            A dictionary of metric results.
        """
        # Add a component axis if not present (as Metric squeezes it out)
        if x.ndim == 0:
            x = x.unsqueeze(dim=0)

        # Same splitting as the original classwise wrapper
        res = {f"{self._prefix}/{lab}": val for lab, val in zip(self.labels, x)}

        # Add in the mean
        res[f"{self._prefix}/mean"] = x.mean(0, dtype=torch.float)

        return res
