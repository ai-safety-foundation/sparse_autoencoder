"""Data parallel utils."""
from typing import Any, Generic, TypeVar

from torch.nn import DataParallel, Module


T = TypeVar("T", bound=Module)


class DataParallelWithModelAttributes(DataParallel[T], Generic[T]):
    """Data parallel with access to underlying model attributes/methods.

    Allows access to underlying model attributes/methods, which is not possible with the default
    `DataParallel` class. Based on:
    https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html

    Example:
        >>> from sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
        >>> model = SparseAutoencoder(SparseAutoencoderConfig(
        ...     n_input_features=2,
        ...     n_learned_features=4,
        ... ))
        >>> distributed_model = DataParallelWithModelAttributes(model)
        >>> distributed_model.config.n_learned_features
        4
    """

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Allow access to underlying model attributes/methods.

        Args:
            name: Attribute/method name.

        Returns:
            Attribute value/method.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
