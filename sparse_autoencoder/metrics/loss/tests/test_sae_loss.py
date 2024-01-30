"""Test the sparse autoencoder loss metric."""
from jaxtyping import Float
import pytest
from torch import Tensor, allclose, ones, rand, tensor, zeros

from sparse_autoencoder.metrics.loss.l1_absolute_loss import L1AbsoluteLoss
from sparse_autoencoder.metrics.loss.l2_reconstruction_loss import L2ReconstructionLoss
from sparse_autoencoder.metrics.loss.sae_loss import SparseAutoencoderLoss
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.parametrize(
    # Each source/decoded tensor is of the form (batch_size, num_components, num_features)
    (
        "source_activations",
        "learned_activations",
        "decoded_activations",
        "l1_coefficient",
        "expected_loss",
    ),
    [
        pytest.param(
            ones(2, 3),
            zeros(2, 4),  # Fully sparse = no activity
            ones(2, 3),
            0.01,
            tensor(0.0),
            id="Perfect reconstruction & perfect sparsity -> zero loss (single component)",
        ),
        pytest.param(
            ones(2, 2, 3),
            zeros(2, 2, 4),
            ones(2, 2, 3),
            0.01,
            tensor([0.0, 0.0]),
            id="Perfect reconstruction & perfect sparsity -> zero loss (2 components)",
        ),
        pytest.param(
            ones(2, 3),
            ones(2, 4),  # Abs error of 1.0 per component => average of 4 loss
            ones(2, 3),
            0.01,
            tensor(0.04),
            id="Just sparsity error (single component)",
        ),
        pytest.param(
            ones(2, 2, 3),
            ones(2, 2, 4),
            ones(2, 2, 3),
            0.01,
            tensor([0.04, 0.04]),
            id="Just sparsity error (2 components)",
        ),
        pytest.param(
            zeros(2, 3),
            zeros(2, 4),
            ones(2, 3),
            0.01,
            tensor(1.0),
            id="Just reconstruction error (single component)",
        ),
        pytest.param(
            zeros(2, 2, 3),
            zeros(2, 2, 4),
            ones(2, 2, 3),
            0.01,
            tensor([1.0, 1.0]),
            id="Just reconstruction error (2 components)",
        ),
        pytest.param(
            zeros(2, 3),
            ones(2, 4),
            ones(2, 3),
            0.01,
            tensor(1.04),
            id="Sparsity and reconstruction error (single component)",
        ),
        pytest.param(
            zeros(2, 2, 3),
            ones(2, 2, 4),
            ones(2, 2, 3),
            0.01,
            tensor([1.04, 1.04]),
            id="Sparsity and reconstruction error (2 components)",
        ),
    ],
)
def test_sae_loss(
    source_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ],
    learned_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
    ],
    decoded_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ],
    l1_coefficient: float,
    expected_loss: Float[Tensor, Axis.COMPONENT_OPTIONAL],
) -> None:
    """Test the SAE loss."""
    num_components: int = source_activations.shape[1] if source_activations.ndim == 3 else 1  # noqa: PLR2004
    metric = SparseAutoencoderLoss(num_components, l1_coefficient)

    res = metric.forward(
        source_activations=source_activations,
        learned_activations=learned_activations,
        decoded_activations=decoded_activations,
    )

    assert allclose(res, expected_loss)


def test_compare_sae_loss_to_composition() -> None:
    """Test the SAE loss metric against composition of l1 and l2."""
    num_components = 3
    l1_coefficient = 0.01
    l1 = L1AbsoluteLoss(num_components)
    l2 = L2ReconstructionLoss(num_components)
    composition_loss = l1 * l1_coefficient + l2

    sae_loss = SparseAutoencoderLoss(num_components, l1_coefficient)

    source_activations = rand(2, num_components, 3)
    learned_activations = rand(2, num_components, 4)
    decoded_activations = rand(2, num_components, 3)

    composition_res = composition_loss.forward(
        source_activations=source_activations,
        learned_activations=learned_activations,
        decoded_activations=decoded_activations,
    )

    sae_res = sae_loss.forward(
        source_activations=source_activations,
        learned_activations=learned_activations,
        decoded_activations=decoded_activations,
    )

    assert allclose(composition_res, sae_res)
