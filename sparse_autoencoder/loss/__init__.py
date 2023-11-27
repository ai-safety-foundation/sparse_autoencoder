"""Loss Modules.

Loss modules are specialised PyTorch modules that calculate the loss for a Sparse Autoencoder. They
all inherit from AbstractLoss, which defines the interface for loss modules and some common methods.

If you want to create your own loss function, see :class:`AbstractLoss`.

For combining multiple loss modules into a single loss module, see :class:`LossReducer`.
"""
