"""Tensor Types.

Tensor types with axis labels. Note that this uses the `jaxtyping` library, which works with PyTorch
tensors as well. Note also that shape sizes are included in the docstrings as well as in the types
as this is needed for IDEs such as VSCode to provide code hints.
"""
from enum import auto
from typing import TypeAlias

from jaxtyping import Float, Int
from strenum import LowercaseStrEnum
from torch import Tensor


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
        >>> batch: TypeAlias = Float[Tensor, Axis.dims(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]
        >>> print(batch)
        <class 'jaxtyping.Float[Tensor, 'batch input_output_feature']'>

        You can also join multiple axis together to represent the dimensions of a tensor:

        >>> print(Axis.dims(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE))
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
    def dims(*axis: "Axis") -> str:
        """Join multiple axis together, to represent the dimensions of a tensor.

        Example:
            >>> print(Axis.dims(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE))
            batch input_output_feature

        Args:
            *axis: Axis to join.

        Returns:
            Joined axis string.
        """
        return " ".join(axis)


# Activation vectors
InputOutputActivationVector: TypeAlias = Float[Tensor, Axis.INPUT_OUTPUT_FEATURE]
"""Input/output activation vector.

This is either a input activation vector from the source model, or a decoded activation vector
from the autoencoder.
"""

LearntActivationVector: TypeAlias = Float[Tensor, Axis.LEARNT_FEATURE]
"""Learned activation vector.

Activation vector from the hidden (learnt) layer of the autoencoder. Typically this is larger than
the input/output activation vector.
"""

# Activation batches/stores
StoreActivations: TypeAlias = Float[Tensor, Axis.dims(Axis.ITEMS, Axis.INPUT_OUTPUT_FEATURE)]
"""Store of activation vectors.

This is used to store large numbers of activation vectors from the source model.
"""

SourceModelActivations: TypeAlias = Float[Tensor, Axis.dims(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)]
"""Source model activations.

Can have any number of proceeding dimensions (e.g. an attention head may generate activations of
shape (batch_size, num_heads, seq_len, feature_dim).
"""

InputOutputActivationBatch: TypeAlias = Float[
    Tensor, Axis.dims(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)
]
"""Input/output activation batch.

This is either a batch of input activation vectors from the source model, or a batch of decoded
activation vectors from the autoencoder.

Shape (batch, input_output_feature)
"""

LearnedActivationBatch: TypeAlias = Float[Tensor, Axis.dims(Axis.BATCH, Axis.LEARNT_FEATURE)]
"""Learned activation batch.

This is a batch of activation vectors from the hidden (learnt) layer of the autoencoder. Typically
the feature dimension is larger than the input/output activation vector.
"""

# Statistics
TrainBatchStatistic: TypeAlias = Float[Tensor, Axis.BATCH]
"""Train batch statistic.

Contains one scalar value per item in the batch.
"""

# Weights and biases
EncoderWeights: TypeAlias = Float[Tensor, Axis.dims(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]
"""Encoder weights.

These weights are part of the encoder module of the autoencoder, responsible for decompressing the
input data (activations from a source model) into a higher-dimensional representation.

The dictionary vectors (basis vectors in the learnt feature space), they can be thought of as
columns of this weight matrix, where each column corresponds to a particular feature in the
lower-dimensional space. The sparsity constraint (hopefully) enforces that they respond relatively
strongly to only a small portion of possible input vectors.

Shape: (learnt_feature_dim, input_output_feature_dim)
"""

DecoderWeights: TypeAlias = Float[Tensor, Axis.dims(Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE)]
"""Decoder weights.

These weights form the decoder part of the autoencoder, which aims to reconstruct the original input
data from the decompressed representation created by the encoder.

Viewing the dictionary vectors in the context of reconstruction, they can be thought of as rows in
this weight matrix.

Shape: (input_output_feature_dim, learnt_feature_dim)
"""

# Weights and biases updated
NeuronActivity: TypeAlias = Int[Tensor, Axis.LEARNT_FEATURE]
"""Neuron activity.

Number of times each neuron has fired (since the last reset).
"""

InputOutputNeuronIndices: TypeAlias = Int[Tensor, Axis.INPUT_OUTPUT_FEATURE]
"""Input/output neuron indices."""

LearntNeuronIndices: TypeAlias = Int[Tensor, Axis.LEARNT_FEATURE_IDX]
"""Learnt neuron indices."""

SampledDeadNeuronInputs: TypeAlias = Float[
    Tensor, Axis.dims(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
]
"""Sampled dead neuron inputs.

Shape: (dead_feature, input_output_feature)
"""

AliveEncoderWeights: TypeAlias = Float[Tensor, Axis.dims(Axis.LEARNT_FEATURE, Axis.ALIVE_FEATURE)]
"""Alive encoder weights."""

DeadEncoderNeuronWeightUpdates: TypeAlias = Float[
    Tensor, Axis.dims(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
]
"""Dead encoder neuron weight updates.

Shape (learnt_feature, dead_feature)
"""

DeadEncoderNeuronBiasUpdates: TypeAlias = Float[Tensor, Axis.DEAD_FEATURE]
"""Dead encoder neuron bias updates."""

DeadDecoderNeuronWeightUpdates: TypeAlias = Float[
    Tensor, Axis.dims(Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)
]
"""Dead decoder neuron weight updates.

Shape (dead_feature, learnt_feature)
"""

# Other
BatchTokenizedPrompts: TypeAlias = Int[Tensor, Axis.dims(Axis.SOURCE_DATA_BATCH, Axis.POSITION)]
"""Batch of tokenized prompts."""

ItemTensor: TypeAlias = Float[Tensor, Axis.SINGLE_ITEM]
"""Single element item tensor."""
