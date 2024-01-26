"""Classwise metrics wrapper."""
import torch
from torch import Tensor
from torchmetrics import ClasswiseWrapper


class ClasswiseWrapperWithMean(ClasswiseWrapper):
    """Classwise wrapper with mean.

    This metric works together with classification metrics that returns multiple values (one value
    per class) such that label information can be automatically included in the output. It extends
    the standard torchmetrics wrapper that does this, adding in an additional mean value (across all
    classes).
    """

    def _convert(self, x: Tensor) -> dict[str, Tensor]:
        """Convert the input tensor to a dictionary of metrics.

        Args:
            x: The input tensor.

        Returns:
            A dictionary of metric results.
        """
        print("here")
        print(self._prefix)
        print(x.shape)

        # Same naming logic as original classwise wrapper
        if not self._prefix and not self._postfix:
            prefix = f"{self.metric.__class__.__name__.lower()}_"
            postfix = ""
        else:
            prefix = self._prefix or ""
            postfix = self._postfix or ""

        # Same splitting as the original classwise wrapper
        if self.labels is None:
            res = {f"{prefix}{i}{postfix}": val for i, val in enumerate(x)}
        else:
            res = {f"{prefix}{lab}{postfix}": val for lab, val in zip(self.labels, x)}

        # add in the mean
        res[f"{prefix}mean{postfix}"] = x.mean(0, dtype=torch.float)
        return res
