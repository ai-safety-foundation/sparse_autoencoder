"""Linear layer with unit norm weights."""
import math

import einops
from jaxtyping import Float
import torch
from torch import Tensor
from torch.nn import Module, init
from torch.nn.parameter import Parameter


class ConstrainedUnitNormLinear(Module):
    """Constrained unit norm linear layer.

    Linear layer for autoencoders, where the dictionary vectors (columns of the weight matrix) are
    constrained to have unit norm. This is done by removing the gradient information parallel to the
    dictionary vectors before applying the gradient step, using a backward hook.

    Motivation:
        Unit norming the dictionary vectors, which are essentially the columns of the encoding and
        decoding matrices, serves a few purposes:

            1. It helps with numerical stability, by preventing the dictionary vectors from growing
                too large.
            2. It acts as a form of regularization, preventing overfitting by not allowing any one
                feature to dominate the representation. It limits the capacity of the model by
                forcing the dictionary vectors to live on the hypersphere of radius 1.
            3. It encourages sparsity. Since the dictionary vectors have a fixed length, the model
                must carefully select which features to activate in order to best reconstruct the
                input.

        Note that the *Towards Monosemanticity: Decomposing Language Models With Dictionary
        Learning* paper found that removing the gradient information parallel to the dictionary
        vectors before applying the gradient step, rather than resetting the dictionary vectors to
        unit norm after each gradient step, results in a small but real reduction in total
        loss](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization).
    """

    in_features: int
    """Number of input features."""

    out_features: int
    """Number of output features."""

    DIMENSION_CONSTRAIN_UNIT_NORM: int = -1
    """Dimension to constrain to unit norm."""

    weight: Float[Tensor, " out_features in_features"]
    """Weight parameter."""

    bias: Float[Tensor, " out_features"] | None
    """Bias parameter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the constrained unit norm linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term.
            device: Device to use.
            dtype: Data type to use.
        """
        # Create the linear layer as per the standard PyTorch linear layer
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Register backward hook to remove any gradient information parallel to the dictionary
        # vectors (columns of the weight matrix) before applying the gradient step.
        self.weight.register_hook(self._weight_backward_hook)

    def reset_parameters(self) -> None:
        """Initialize or reset the parameters.

        Example:
            >>> import torch
            >>> layer = ConstrainedUnitNormLinear(3, 3)
            >>> layer.reset_parameters()
            >>> column_norms = torch.sum(layer.weight ** 2, dim=1)
            >>> column_norms.round(decimals=3).tolist()
            [1.0, 1.0, 1.0]

        """
        # Initialize the weights with a normal distribution. Note we don't use e.g. kaiming
        # normalisation here, since we immediately scale the weights to have unit norm (so the
        # initial standard deviation doesn't matter). Note also that `init.normal_` is in place.
        self.weight: Float[Tensor, "out_features in_features"] = init.normal_(
            self.weight, mean=0, std=1
        )

        # Scale so that each column has unit norm
        with torch.no_grad():
            torch.nn.functional.normalize(
                self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM, out=self.weight
            )

        # Initialise the bias
        # This is the standard approach used in `torch.nn.Linear.reset_parameters`
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _weight_backward_hook(
        self,
        grad: Float[Tensor, "out_features in_features"],
    ) -> Float[Tensor, "out_features in_features"]:
        """Unit norm backward hook.

        By subtracting the projection of the gradient onto the dictionary vectors, we remove the
        component of the gradient that is parallel to the dictionary vectors and just keep the
        component that is orthogonal to the dictionary vectors (i.e. moving around the hypersphere).
        The result is that the backward pass does not change the norm of the dictionary vectors.

        Example:
            >>> import torch
            >>> layer = ConstrainedUnitNormLinear(3, 4)
            >>> data = torch.randn((3), requires_grad=True)
            >>> logits = layer(data)
            >>> loss = torch.sum(logits ** 2)
            >>> loss.backward() # The hook is applied here
            >>> weight_norms = torch.sum(layer.weight.data ** 2, dim=1)
            >>> weight_norms.round(decimals=3).tolist()
            [1.0, 1.0, 1.0, 1.0]

        Args:
            grad: Gradient with respect to the weights.
        """
        # Project the gradients onto the dictionary vectors. Intuitively the dictionary vectors can
        # be thought of as vectors that end on the circumference of a hypersphere. The projection of
        # the gradient onto the dictionary vectors is the component of the gradient that is parallel
        # to the dictionary vectors, i.e. the component that moves to or from the center of the
        # hypersphere.
        dot_product: Float[Tensor, " out_features"] = einops.einsum(
            grad, self.weight, "out_features in_features, out_features in_features -> out_features"
        )

        normalized_weight: Float[Tensor, "out_features in_features"] = self.weight / torch.norm(
            self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM, keepdim=True
        )

        projection = einops.einsum(
            dot_product,
            normalized_weight,
            "out_features, out_features in_features -> out_features in_features",
        )

        # Subtracting the parallel component from the gradient leaves only the component that is
        # orthogonal to the dictionary vectors, i.e. the component that moves around the surface of
        # the hypersphere.
        return grad - projection

    def constrain_weights_unit_norm(self) -> None:
        """Constrain the weights to have unit norm.

        Note this must be called after each gradient step. This is because optimisers such as Adam
        don't strictly follow the gradient, but instead follow a modified gradient that includes
        momentum. This means that the gradient step can change the norm of the dictionary vectors,
        even when the hook :meth:`_weight_backward_hook` is applied.

        Note this can't be applied directly in the backward hook, as it would interfere with a
        variety of use cases (e.g. gradient accumulation across mini-batches, concurrency issues
        with asynchronous operations, etc).

        Example:
            >>> import torch
            >>> layer = ConstrainedUnitNormLinear(3, 3)
            >>> layer.weight.data = torch.ones((3, 3)) * 10
            >>> layer.constrain_weights_unit_norm()
            >>> column_norms = torch.sum(layer.weight ** 2, dim=1)
            >>> column_norms.round(decimals=3).tolist()
            [1.0, 1.0, 1.0]

        """
        with torch.no_grad():
            torch.nn.functional.normalize(
                self.weight, dim=self.DIMENSION_CONSTRAIN_UNIT_NORM, out=self.weight
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output of the forward pass.
        """
        # Prevent the drift of the dictionary vectors away from unit norm. This can happen even
        # though we remove the gradient information parallel to the dictionary vectors before
        # applying the gradient step, since optimisers such as Adam don't strictly follow the
        # gradient, but instead follow a modified gradient that includes momentum.
        self.constrain_weights_unit_norm()

        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        """String extra representation of the module."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
