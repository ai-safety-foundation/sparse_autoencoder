"""Test the L1 absolute loss metric."""
from jaxtyping import Float
import pytest
from torch import Tensor, allclose, ones, tensor, zeros

from sparse_autoencoder.metrics.loss.l1_absolute_loss import L1AbsoluteLoss
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.parametrize(
    # Each source/decoded tensor is of the form (batch_size, num_components, num_features)
    ("learned_activations", "expected_loss"),
    [
        pytest.param(
            zeros(2, 3),
            tensor(0.0),
            id="All zero activations -> zero loss (single component)",
        ),
        pytest.param(
            zeros(2, 2, 3),
            tensor([0.0, 0.0]),
            id="All zero activations -> zero loss (2 components)",
        ),
        pytest.param(
            ones(2, 3),  # 3 features -> 3.0 loss
            tensor(3.0),
            id="All 1.0 activations -> 3.0 loss (single component)",
        ),
        pytest.param(
            ones(2, 2, 3),
            tensor([3.0, 3.0]),
            id="All 1.0 activations -> 3.0 loss (2 components)",
        ),
        pytest.param(
            ones(2, 2, 3) * -1,  # Loss is absolute so the same as +ve 1s
            tensor([3.0, 3.0]),
            id="All -ve 1.0 activations -> 3.0 loss (2 components)",
        ),
    ],
)
def test_l1_absolute_loss(
    learned_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
    ],
    expected_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL],
) -> None:
    """Test the L1 absolute loss."""
    num_components: int = learned_activations.shape[1] if learned_activations.ndim == 3 else 1  # noqa: PLR2004
    l1 = L1AbsoluteLoss(num_components)

    res = l1.forward(learned_activations=learned_activations)

    assert allclose(res, expected_loss)
