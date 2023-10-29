"""Tied Bias Tests."""
import torch

from sparse_autoencoder.autoencoder.tied_bias import PostEncoderBias, PreEncoderBias


def test_pre_encoder_subtracts_bias() -> None:
    """Check that the pre-encoder bias subtracts the bias."""
    encoder_input = torch.tensor([5.0, 3.0, 1.0])
    bias = torch.tensor([2.0, 4.0, 6.0])
    expected = encoder_input - bias

    pre_encoder = PreEncoderBias(bias)
    output = pre_encoder(encoder_input)

    assert torch.allclose(output, expected)


def test_post_encoder_adds_bias() -> None:
    """Check that the post-encoder bias adds the bias."""
    decoder_output = torch.tensor([5.0, 3.0, 1.0])
    bias = torch.tensor([2.0, 4.0, 6.0])
    expected = decoder_output + bias

    post_encoder = PostEncoderBias(bias)
    output = post_encoder(decoder_output)

    assert torch.allclose(output, expected)
