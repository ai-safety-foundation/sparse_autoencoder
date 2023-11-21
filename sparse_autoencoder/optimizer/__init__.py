"""Optimizers for Sparse Autoencoders.

When training a Sparse Autoencoder, it can be necessary to manually edit the model parameters
(e.g. with neuron resampling to prevent dead neurons). When doing this, it's also necessary to
reset the optimizer state for these parameters, as otherwise things like running averages will be
incorrect (e.g. the running averages of the gradients and the squares of gradients with Adam).

The optimizer used in the original [Towards Monosemanticity: Decomposing Language Models With
Dictionary Learning](Towards Monosemanticity: Decomposing Language Models With Dictionary Learning)
paper is available here as :class:`AdamWithReset`.

To enable creating other optimizers with reset methods, we also provide the interface
:class:`AbstractOptimizerWithReset`.
"""

from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset


__all__ = ["AdamWithReset", "AbstractOptimizerWithReset"]
