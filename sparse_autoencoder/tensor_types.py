"""Tensor Types.

Tensor types with axis labels.
"""
from enum import auto
from typing import TypeAlias

from jaxtyping import Float, Int
from strenum import LowercaseStrEnum
from torch import Tensor


class Axis(LowercaseStrEnum):
    """Tensor axis names.

    Example:
        Can be used directly.

        >>> print(Axis.TRAIN_FEATURE)
        train_feature
    """

    # Batches
    SOURCE_BATCH = auto()
    """Source data batch (e.g. batch of prompts)."""

    GENERATED_BATCH = auto()
    """Generated batch."""

    TRAIN_BATCH = auto()
    """Train batch (e.g. batch of activations from the source model that are being trained on)."""

    VALIDATION_BATCH = auto()
    """Validation batch (e.g. batch of learned activations)."""

    ITEMS = auto()
    """Arbitrary number of items."""

    # Features

    TRAIN_FEATURE = auto()
    """Training feature (e.g. feature in input activation vector)."""

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


InputActivationsStatistic: TypeAlias = Float[Tensor, Axis.TRAIN_FEATURE]
"""Input activation statistic."""

TokenizedSourceDataBatch: TypeAlias = Int[Tensor, Axis.SOURCE_BATCH + " " + Axis.POSITION]
"""Tokenized source data batch."""

SourceModelActivations: TypeAlias = Float[Tensor, Axis.ANY + " " + Axis.TRAIN_FEATURE]
"""Source model activations."""

GeneratedActivation: TypeAlias = Float[Tensor, Axis.TRAIN_FEATURE]
"""Generated activation."""

GeneratedActivationBatch: TypeAlias = Float[Tensor, Axis.GENERATED_BATCH + " " + Axis.TRAIN_FEATURE]
"""Generated activation batch."""

GeneratedActivationStore: TypeAlias = Float[Tensor, Axis.ITEMS + " " + Axis.TRAIN_FEATURE]
"""Generated activation tensor store."""

LearnedActivationBatch: TypeAlias = Float[Tensor, Axis.TRAIN_BATCH + " " + Axis.LEARNT_FEATURE]
"""Learned activation batch."""

SourceActivationBatch: TypeAlias = Float[Tensor, Axis.TRAIN_BATCH + " " + Axis.TRAIN_FEATURE]
"""Source (input) activation batch."""

DecodedActivationBatch: TypeAlias = Float[Tensor, Axis.TRAIN_BATCH + " " + Axis.TRAIN_FEATURE]
"""Decoded activation batch."""

ValidationActivationBatch: TypeAlias = Float[
    Tensor, Axis.VALIDATION_BATCH + " " + Axis.LEARNT_FEATURE
]
"""Validation activation batch."""

ValidationBatch: TypeAlias = Float[Tensor, Axis.VALIDATION_BATCH]
"""Validation batch."""

EncoderWeights: TypeAlias = Float[Tensor, Axis.LEARNT_FEATURE + " " + Axis.TRAIN_FEATURE]
"""Encoder weights."""

LearnedFeatures: TypeAlias = Float[Tensor, Axis.LEARNT_FEATURE]
"""Learned features."""

DecoderWeights: TypeAlias = Float[Tensor, Axis.TRAIN_FEATURE + " " + Axis.LEARNT_FEATURE]
"""Decoder weights."""

DecoderBias: TypeAlias = Float[Tensor, Axis.TRAIN_FEATURE]
"""Decoder bias."""

BatchItemwiseLoss: TypeAlias = Float[Tensor, Axis.TRAIN_BATCH]
"""Batch itemwise loss"""

NeuronActivity: TypeAlias = Int[Tensor, Axis.LEARNT_FEATURE]
"""Neuron activity.

Number of times each neuron has fired (since the last reset).
"""

DeadNeuronIndices: TypeAlias = Int[Tensor, Axis.LEARNT_FEATURE_IDX]
"""Dead neuron indices."""

SampledDeadNeuronInputs: TypeAlias = Float[Tensor, Axis.DEAD_FEATURE + " " + Axis.TRAIN_FEATURE]
"""Sampled dead neuron inputs."""

AliveEncoderWeights: TypeAlias = Float[Tensor, Axis.LEARNT_FEATURE + " " + Axis.ALIVE_FEATURE]
"""Alive encoder weights."""

DeadEncoderNeuronWeightUpdates: TypeAlias = Float[
    Tensor, Axis.DEAD_FEATURE + " " + Axis.TRAIN_FEATURE
]
"""Dead encoder neuron weight updates."""

DeadEncoderNeuronBiasUpdates: TypeAlias = Float[Tensor, Axis.DEAD_FEATURE]
"""Dead encoder neuron bias updates."""

DeadDecoderNeuronWeightUpdates: TypeAlias = Float[
    Tensor, Axis.TRAIN_FEATURE + " " + Axis.DEAD_FEATURE
]
"""Dead decoder neuron weight updates."""

ItemTensor: TypeAlias = Float[Tensor, Axis.SINGLE_ITEM]
"""Single element item tensor."""
