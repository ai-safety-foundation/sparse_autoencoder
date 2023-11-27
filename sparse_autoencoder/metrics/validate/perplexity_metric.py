"""Doing a periodic evaluation of the autoencoder."""
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.autoencoder.abstract_autoencoder import \
    AbstractAutoencoder
from sparse_autoencoder.metrics.validate.abstract_validate_metric import (
    AbstractValidationMetric, ValidationMetricContext)
from sparse_autoencoder.train.utils import get_model_device

N_BATCHES = 100


def make_autoencoder_hook(
    autoencoder: AbstractAutoencoder
) -> Callable[[Tensor, HookPoint], Tensor]:
    """Make the hook that will go into the forward pass of the model."""

    def autoencoder_hook(input_activations: Tensor, hook: HookPoint) -> Tensor:  # noqa: ARG001
        """Hook to get the activations from the autoencoder."""
        to_sae_activations = input_activations.reshape(-1, autoencoder.n_input_features)
        _, reconstructed_activations = autoencoder(to_sae_activations)
        return reconstructed_activations.reshape(input_activations.shape)

    return autoencoder_hook


class PerplexityMetric(AbstractValidationMetric):
    """Calculate the change in perplexity when using the autoencoder in the original model."""

    def calculate(self, context: ValidationMetricContext) -> dict[str, Any]:
        """Run a perplexity evaluation of the model with and without the autoencoder."""
        total_standard_loss = 0.0
        total_sae_loss = 0.0
        total_kl_divergence = 0.0
        batches = 0

        device = get_model_device(context.source_model)

        with torch.no_grad():
            for batch_raw in context.dataset:
                batch = torch.tensor(batch_raw["input_ids"]).to(device=device).to(torch.int64)[:2]
                print(batch.shape)

                # Get the activations from the model
                sae_hook = make_autoencoder_hook(context.autoencoder)
                # standard_loss2 = context.source_model.run_with_hooks(batch, return_type="loss")
                # sae_loss2 = context.source_model.run_with_hooks(
                #     batch, return_type="loss", fwd_hooks=[(context.hook_point, sae_hook)]
                # )

                standard_logits = context.source_model.run_with_hooks(batch, return_type="logits")
                sae_logits = context.source_model.run_with_hooks(
                    batch, return_type="logits", fwd_hooks=[(context.hook_point, sae_hook)]
                )

                kl_divergence = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(standard_logits, dim=-1),
                    torch.nn.functional.softmax(sae_logits, dim=-1),
                    reduction="batchmean",
                )
                kl_divergence = kl_divergence / batch.shape[0]
                # loss should be equal to the negative log likelihood
                sae_logit_outputs = sae_logits[:, :-1].reshape(-1, sae_logits.shape[-1])
                standard_logit_outputs = standard_logits[:, :-1].reshape(-1, standard_logits.shape[-1])
                # sae_log_probs = torch.nn.functional.log_softmax(sae_logit_outputs, dim=-1)
                standard_log_probs = torch.nn.functional.log_softmax(standard_logit_outputs, dim=-1)

                targets = batch[1:]
                criterion = torch.nn.CrossEntropyLoss()
                nll_loss = torch.nn.NLLLoss()
                targets2 = torch.tensor(targets[0].item()).to(device=device).to(torch.int64).unsqueeze(0)
                print(criterion(sae_logit_outputs, targets2))
                print(criterion(sae_logit_outputs, targets))

                sae_loss = criterion(sae_logit_outputs, targets)
                standard_loss = torch.nn.functional.nll_loss(
                    standard_log_probs, targets, reduction="mean"
                )
                breakpoint()

                total_standard_loss += standard_loss.item()
                total_sae_loss += sae_loss.item()

                total_kl_divergence += kl_divergence.item()

                batches += 1
                if batches >= N_BATCHES:
                    break

        return {
            "Standard loss": total_standard_loss / batches,
            "SAE loss": total_sae_loss / batches,
            "KL divergence": total_kl_divergence / batches,
        }
