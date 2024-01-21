"""Test the L2 reconstruction loss metric."""
from jaxtyping import Float
import pytest
from torch import Tensor, allclose, ones, tensor, zeros

from sparse_autoencoder.metrics.loss.l2_reconstruction_loss import L2ReconstructionLoss
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.parametrize(
    # Each source/decoded tensor is of the form (batch_size, num_components, num_features)
    ("source_activations", "decoded_activations", "expected_loss"),
    [
        pytest.param(
            ones(2, 3),
            ones(2, 3),
            tensor(0.0),
            id="Perfect reconstruction -> zero loss (single component)",
        ),
        pytest.param(
            ones(2, 2, 3),
            ones(2, 2, 3),
            tensor([0.0, 0.0]),
            id="Perfect reconstruction -> zero loss (2 components)",
        ),
        pytest.param(
            zeros(2, 3),
            ones(2, 3),
            tensor(1.0),
            id="All errors 1.0 -> 1.0 loss (single component)",
        ),
        pytest.param(
            zeros(2, 2, 3),
            ones(2, 2, 3),
            tensor([1.0, 1.0]),
            id="All errors 1.0 -> 1.0 loss (2 components)",
        ),
    ],
)
def test_l2_reconstruction_loss(
    source_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ],
    decoded_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ],
    expected_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL],
) -> None:
    """Test the L2 reconstruction loss."""
    num_components: int = source_activations.shape[1] if source_activations.ndim == 3 else 1  # noqa: PLR2004
    l2 = L2ReconstructionLoss(num_components)

    res = l2.forward(decoded_activations=decoded_activations, source_activations=source_activations)

    assert allclose(res, expected_loss)
