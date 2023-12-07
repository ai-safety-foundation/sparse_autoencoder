"""Tensor Axis Types."""
from enum import auto

from strenum import LowercaseStrEnum


class Axis(LowercaseStrEnum):
    """Tensor axis names.

    Used to annotate tensor types.

    Example:
        When used directly it prints a string:

        >>> print(Axis.INPUT_OUTPUT_FEATURE)
        input_output_feature

        The primary use is to annotate tensor types:

        >>> from jaxtyping import Float
        >>> from torch import Tensor
        >>> from typing import TypeAlias
        >>> batch: TypeAlias = Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]
        >>> print(batch)
        <class 'jaxtyping.Float[Tensor, 'batch input_output_feature']'>

        You can also join multiple axis together to represent the dimensions of a tensor:

        >>> print(Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE))
        batch input_output_feature
    """

    # Batches
    SOURCE_DATA_BATCH = auto()
    """Batch of prompts used to generate source model activations."""

    BATCH = auto()
    """Batch of items that the SAE is being trained on."""

    ITEMS = auto()
    """Arbitrary number of items."""

    # Features
    INPUT_OUTPUT_FEATURE = auto()
    """Input or output feature (e.g. feature in activation vector from source model)."""

    LEARNT_FEATURE = auto()
    """Learn feature (e.g. feature in learnt activation vector)."""

    DEAD_FEATURE = auto()
    """Dead feature."""

    ALIVE_FEATURE = auto()
    """Alive feature."""

    # Feature indices
    LEARNT_FEATURE_IDX = auto()

    # Other
    POSITION = auto()
    """Token position."""

    SINGLE_ITEM = ""
    """Single item axis."""

    ANY = "*any"
    """Any number of axis."""

    @staticmethod
    def names(*axis: "Axis") -> str:
        """Join multiple axis together, to represent the dimensions of a tensor.

        Example:
            >>> print(Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE))
            batch input_output_feature

        Args:
            *axis: Axis to join.

        Returns:
            Joined axis string.
        """
        return " ".join(a.value for a in axis)
