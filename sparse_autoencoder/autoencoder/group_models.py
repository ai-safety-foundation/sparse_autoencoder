"""Group of sparse autoencoders."""
from jaxtyping import Float
from torch import Tensor, nn

from sparse_autoencoder.autoencoder.abstract_autoencoder import AbstractAutoencoder
from sparse_autoencoder.tensor_types import Axis


class GroupAutoencoderModels(nn.Module):
    """Group of sparse autoencoders."""

    def __init__(self, models: list[AbstractAutoencoder]):
        """Initialize the group of sparse autoencoders."""
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(
        self,
        x: list[Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]],
    ) -> list[
        tuple[
            Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)],
            Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)],
        ]
    ]:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from all residual stream layers in a transformer
                model).

        Returns:
            Tuple of learned activations and decoded activations.
        """
        return [model(x) for model in self.models]
