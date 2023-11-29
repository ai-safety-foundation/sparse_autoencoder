"""Compare the SAE implementation to Neel's 1L Implementation.

https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py
"""
import torch
from torch import nn

from sparse_autoencoder.autoencoder.model import SparseAutoencoder


class NeelAutoencoder(nn.Module):
    """Neel's 1L autoencoder implementation."""

    def __init__(
        self,
        d_hidden: int,
        act_size: int,
        l1_coeff: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the autoencoder."""
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(act_size, dtype=dtype))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(act_size, d_hidden, dtype=dtype))
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, act_size, dtype=dtype))
        )

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x_cent = x - self.b_dec
        acts = nn.functional.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Make decoder weights and gradient unit norm."""
        with torch.no_grad():
            weight_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
            weight_dec_grad_proj = (self.W_dec.grad * weight_dec_normed).sum(
                -1, keepdim=True
            ) * weight_dec_normed
            self.W_dec.grad -= weight_dec_grad_proj
            # Bugfix(?)
            self.W_dec.data = weight_dec_normed


def test_biases_initialised_same_way() -> None:
    """Test that the biases are initialised the same."""
    n_input_features: int = 2
    n_learned_features: int = 3
    l1_coefficient: float = 0.01

    torch.random.manual_seed(0)
    autoencoder = SparseAutoencoder(
        n_input_features=n_input_features,
        n_learned_features=n_learned_features,
    )

    torch.random.manual_seed(0)
    neel_autoencoder = NeelAutoencoder(
        d_hidden=n_learned_features,
        act_size=n_input_features,
        l1_coeff=l1_coefficient,
    )

    assert torch.allclose(autoencoder.tied_bias, neel_autoencoder.b_dec)
    # Note we can't compare weights as Neel's implementation uses rotated tensors and applies
    # kaiming incorrectly (uses leaky relu version and incorrect fan strategy for the rotation
    # used). Note also that the encoder bias is initialised to zero in Neel's implementation,
    # whereas we use the standard PyTorch initialisation.


def test_forward_pass_same_weights() -> None:
    """Test a forward pass with the same weights."""
    n_input_features: int = 12
    n_learned_features: int = 48
    l1_coefficient: float = 0.01

    autoencoder = SparseAutoencoder(
        n_input_features=n_input_features,
        n_learned_features=n_learned_features,
    )
    neel_autoencoder = NeelAutoencoder(
        d_hidden=n_learned_features,
        act_size=n_input_features,
        l1_coeff=l1_coefficient,
    )

    # Set the same weights
    autoencoder.encoder.weight.data = neel_autoencoder.W_enc.data.T
    autoencoder.decoder.weight.data = neel_autoencoder.W_dec.data.T
    autoencoder.tied_bias.data = neel_autoencoder.b_dec.data
    autoencoder.encoder.bias.data = neel_autoencoder.b_enc.data

    # Create some test data
    test_batch = torch.randn(4, n_input_features)
    learned, hidden = autoencoder.forward(test_batch)
    _loss, x_reconstruct, acts, _l2_loss, _l1_loss = neel_autoencoder.forward(test_batch)

    assert torch.allclose(learned, acts)
    assert torch.allclose(hidden, x_reconstruct)


def test_unit_norm_weights() -> None:
    """Test that the decoder weights are unit normalized in the same way."""
    n_input_features: int = 2
    n_learned_features: int = 4
    l1_coefficient: float = 0.01

    autoencoder = SparseAutoencoder(
        n_input_features=n_input_features,
        n_learned_features=n_learned_features,
    )
    neel_autoencoder = NeelAutoencoder(
        d_hidden=n_learned_features,
        act_size=n_input_features,
        l1_coeff=l1_coefficient,
    )
    pre_unit_norm_weights = autoencoder.decoder.weight.clone()
    pre_unit_norm_neel_weights = neel_autoencoder.W_dec.clone()

    # Set the same decoder weights
    decoder_weights = torch.rand_like(autoencoder.decoder.weight)
    autoencoder.decoder._weight.data = decoder_weights  # noqa: SLF001 # type: ignore
    neel_autoencoder.W_dec.data = decoder_weights.T

    # Do a forward & backward pass so we have gradients
    test_batch = torch.randn(4, n_input_features)
    _learned, decoded = autoencoder.forward(test_batch)
    decoded.sum().backward()
    decoded = neel_autoencoder.forward(test_batch)[1]
    decoded.sum().backward()

    # Apply the unit norm
    autoencoder.decoder.constrain_weights_unit_norm()
    neel_autoencoder.make_decoder_weights_and_grad_unit_norm()

    # Check the decoder weights are the same with both models
    assert torch.allclose(autoencoder.decoder.weight, neel_autoencoder.W_dec.T)

    # Check the trivial case that the weights haven't just stayed the same as before the unit norm
    assert not torch.allclose(autoencoder.decoder.weight, pre_unit_norm_weights)
    assert not torch.allclose(neel_autoencoder.W_dec, pre_unit_norm_neel_weights)


def test_unit_norm_weights_grad() -> None:
    """Test that the decoder weights are unit normalized in the same way."""
    torch.random.manual_seed(42)
    n_input_features: int = 2
    n_learned_features: int = 4
    l1_coefficient: float = 0.01

    autoencoder = SparseAutoencoder(
        n_input_features=n_input_features,
        n_learned_features=n_learned_features,
    )
    neel_autoencoder = NeelAutoencoder(
        d_hidden=n_learned_features,
        act_size=n_input_features,
        l1_coeff=l1_coefficient,
    )

    # Set the same decoder weights
    decoder_weights = torch.rand_like(autoencoder.decoder.weight)
    autoencoder.decoder._weight.data = decoder_weights  # noqa: SLF001 # type: ignore
    neel_autoencoder.W_dec.data = decoder_weights.T
    autoencoder.decoder._weight.grad = torch.zeros_like(autoencoder.decoder.weight)  # noqa: SLF001 # type: ignore
    neel_autoencoder.W_dec.grad = torch.zeros_like(neel_autoencoder.W_dec)

    # Set the same tied bias weights
    neel_autoencoder.b_dec.data = autoencoder.tied_bias.data
    neel_autoencoder.b_enc.data = autoencoder.encoder.bias.data
    neel_autoencoder.W_enc.data = autoencoder.encoder.weight.data.T

    # Do a forward & backward pass so we have gradients
    test_batch = torch.randn(4, n_input_features)
    _learned, decoded = autoencoder.forward(test_batch)
    _loss = decoded.sum().backward()
    neel_decoded = neel_autoencoder.forward(test_batch)[1]
    _loss_neel = neel_decoded.sum().backward()

    # Apply the unit norm
    autoencoder.decoder.constrain_weights_unit_norm()
    neel_autoencoder.make_decoder_weights_and_grad_unit_norm()

    # Check the gradient weights are the same
    assert autoencoder.decoder.weight.grad is not None
    assert neel_autoencoder.W_dec.grad is not None
    assert torch.allclose(autoencoder.decoder.weight.grad, neel_autoencoder.W_dec.grad.T, rtol=1e-4)
