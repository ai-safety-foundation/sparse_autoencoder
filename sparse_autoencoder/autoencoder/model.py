"""The Sparse Autoencoder Model."""
from pathlib import Path
from tempfile import gettempdir
from typing import NamedTuple

from huggingface_hub import HfApi, hf_hub_download
from jaxtyping import Float
from pydantic import (
    BaseModel,
    DirectoryPath,
    NonNegativeInt,
    PositiveInt,
    validate_call,
)
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.serialization import FILE_LIKE
import wandb

from sparse_autoencoder.autoencoder.components.linear_encoder import LinearEncoder
from sparse_autoencoder.autoencoder.components.tied_bias import TiedBias, TiedBiasPosition
from sparse_autoencoder.autoencoder.components.unit_norm_decoder import UnitNormDecoder
from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


class SparseAutoencoderConfig(BaseModel, frozen=True):
    """SAE model config."""

    n_input_features: PositiveInt
    """Number of input features.

    E.g. `d_mlp` if training on MLP activations from TransformerLens).
    """

    n_learned_features: PositiveInt
    """Number of learned features.

    The initial paper experimented with 1 to 256 times the number of input features, and primarily
    used a multiple of 8."""

    n_components: PositiveInt | None = None
    """Number of source model components the SAE is trained on.""

    This is useful if you want to train the SAE on several components of the source model at once.
    If `None`, the SAE is assumed to be trained on just one component (in this case the model won't
    contain a component axis in any of the parameters).
    """


class SparseAutoencoderState(BaseModel, arbitrary_types_allowed=True):
    """SAE model state.

    Used for saving and loading the model.
    """

    config: SparseAutoencoderConfig
    """Model config."""

    state_dict: dict[str, Tensor]
    """Model state dict."""


class ForwardPassResult(NamedTuple):
    """SAE model forward pass result."""

    learned_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
    ]

    decoded_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ]


DEFAULT_TMP_DIR = Path(gettempdir()) / "sparse_autoencoder"


class SparseAutoencoder(Module):
    """Sparse Autoencoder Model."""

    config: SparseAutoencoderConfig
    """Model config."""

    geometric_median_dataset: Float[
        Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Estimated Geometric Median of the Dataset.

    Used for initialising :attr:`tied_bias`.
    """

    tied_bias: Float[
        Parameter, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Tied Bias Parameter.

    The same bias is used pre-encoder and post-decoder.
    """

    pre_encoder_bias: TiedBias
    """Pre-Encoder Bias."""

    encoder: LinearEncoder
    """Encoder."""

    decoder: UnitNormDecoder
    """Decoder."""

    post_decoder_bias: TiedBias
    """Post-Decoder Bias."""

    def __init__(
        self,
        config: SparseAutoencoderConfig,
        geometric_median_dataset: Float[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ]
        | None = None,
    ) -> None:
        """Initialize the Sparse Autoencoder Model.

        Args:
            config: Model config.
            geometric_median_dataset: Estimated geometric median of the dataset.
        """
        super().__init__()

        self.config = config

        # Store the geometric median of the dataset (so that we can reset parameters). This is not a
        # parameter itself (the tied bias parameter is used for that), so gradients are disabled.
        tied_bias_shape = shape_with_optional_dimensions(
            config.n_components, config.n_input_features
        )
        if geometric_median_dataset is not None:
            self.geometric_median_dataset = geometric_median_dataset.clone()
            self.geometric_median_dataset.requires_grad = False
        else:
            self.geometric_median_dataset = torch.zeros(tied_bias_shape)
            self.geometric_median_dataset.requires_grad = False

        # Initialize the tied bias
        self.tied_bias = Parameter(torch.empty(tied_bias_shape))
        self.initialize_tied_parameters()

        # Initialize the components
        self.pre_encoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.PRE_ENCODER)

        self.encoder = LinearEncoder(
            input_features=config.n_input_features,
            learnt_features=config.n_learned_features,
            n_components=config.n_components,
        )

        self.decoder = UnitNormDecoder(
            learnt_features=config.n_learned_features,
            decoded_features=config.n_input_features,
            n_components=config.n_components,
        )

        self.post_decoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.POST_DECODER)

    def forward(
        self,
        x: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> ForwardPassResult:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Tuple of learned activations and decoded activations.
        """
        x = self.pre_encoder_bias(x)
        learned_activations = self.encoder(x)
        x = self.decoder(learned_activations)
        decoded_activations = self.post_decoder_bias(x)

        return ForwardPassResult(learned_activations, decoded_activations)

    def initialize_tied_parameters(self) -> None:
        """Initialize the tied parameters."""
        # The tied bias is initialised as the geometric median of the dataset
        self.tied_bias.data = self.geometric_median_dataset

    def reset_parameters(self) -> None:
        """Reset the parameters."""
        self.initialize_tied_parameters()
        for module in self.network:
            if "reset_parameters" in dir(module):
                module.reset_parameters()

    @property
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """
        return (
            self.encoder.reset_optimizer_parameter_details
            + self.decoder.reset_optimizer_parameter_details
        )

    def post_backwards_hook(self) -> None:
        """Hook to be called after each learning step.

        This can be used to e.g. constrain weights to unit norm.
        """
        self.decoder.constrain_weights_unit_norm()

    @staticmethod
    @validate_call
    def get_single_component_state_dict(
        state: SparseAutoencoderState, component_idx: NonNegativeInt
    ) -> dict[str, Tensor]:
        """Get the state dict for a single component.

        Args:
            state: Sparse Autoencoder state.
            component_idx: Index of the component to get the state dict for.

        Returns:
            State dict for the component.

        Raises:
            ValueError: If the state dict doesn't contain a components dimension.
        """
        # Check the state has a components dimension
        if state.config.n_components is None:
            error_message = (
                "Trying to load a single component from the state dict, but the state dict "
                "doesn't contain a components dimension."
            )
            raise ValueError(error_message)

        # Return the state dict for the component
        return {key: value[component_idx] for key, value in state.state_dict.items()}

    def save(self, file_path: Path) -> None:
        """Save the model config and state dict to a file.

        Args:
            file_path: Path to save the model to.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        state = SparseAutoencoderState(config=self.config, state_dict=self.state_dict())
        torch.save(state, file_path)

    @staticmethod
    def load(
        file_path: FILE_LIKE,
        component_idx: PositiveInt | None = None,
    ) -> "SparseAutoencoder":
        """Load the model from a file.

        Args:
            file_path: Path to load the model from.
            component_idx: If loading a state dict from a model that has been trained on multiple
                components (e.g. all MLP layers) you may want to to load just one component. In this
                case you can set `component_idx` to the index of the component to load. Note you
                should not set this if you want to load a state dict from a model that has been
                trained on a single component (or if you want to load all components).

        Returns:
            The loaded model.
        """
        # Load the file
        serialized_state = torch.load(file_path)
        state = SparseAutoencoderState.model_validate(serialized_state)

        # Initialise the model
        config = SparseAutoencoderConfig(
            n_input_features=state.config.n_input_features,
            n_learned_features=state.config.n_learned_features,
            n_components=state.config.n_components if component_idx is None else None,
        )
        state_dict = (
            SparseAutoencoder.get_single_component_state_dict(state, component_idx)
            if component_idx is not None
            else state.state_dict
        )
        model = SparseAutoencoder(config)
        model.load_state_dict(state_dict)

        return model

    def save_to_wandb(
        self,
        artifact_name: str,
        directory: DirectoryPath = DEFAULT_TMP_DIR,
    ) -> str:
        """Save the model to wandb.

        Args:
            artifact_name: A human-readable name for this artifact, which is how you can identify
                this artifact in the UI or reference it in use_artifact calls. Names can contain
                letters, numbers, underscores, hyphens, and dots. The name must be unique across a
                project. Example: "sweep_name 1e9 activations".
            directory: Directory to save the model to.

        Returns:
            Name of the wandb artifact.

        Raises:
            ValueError: If wandb is not initialised.
        """
        # Save the file
        directory.mkdir(parents=True, exist_ok=True)
        file_name = artifact_name + ".pt"
        file_path = directory / file_name
        self.save(file_path)

        # Upload to wandb
        if wandb.run is None:
            error_message = "Trying to save the model to wandb, but wandb is not initialised."
            raise ValueError(error_message)
        artifact = wandb.Artifact(
            artifact_name,
            type="model",
            description="Sparse Autoencoder model state, created with `sparse_autoencoder`.",
        )
        artifact.add_file(str(file_path), name="sae-model-state.pt")
        artifact.save()
        wandb.log_artifact(artifact)
        artifact.wait()

        return artifact.source_qualified_name

    @staticmethod
    def load_from_wandb(
        wandb_artifact_name: str,
        component_idx: PositiveInt | None = None,
    ) -> "SparseAutoencoder":
        """Load the model from wandb.

        Args:
            wandb_artifact_name: Name of the wandb artifact to load the model from (e.g.
                "username/project/artifact_name:version").
            component_idx: If loading a state dict from a model that has been trained on multiple
                components (e.g. all MLP layers) you may want to to load just one component. In this
                case you can set `component_idx` to the index of the component to load. Note you
                should not set this if you want to load a state dict from a model that has been
                trained on a single component (or if you want to load all components).

        Returns:
            The loaded model.
        """
        api = wandb.Api()
        artifact = api.artifact(wandb_artifact_name, type="model")
        download_path = artifact.download()
        return SparseAutoencoder.load(Path(download_path) / "sae-model-state.pt", component_idx)

    def save_to_hugging_face(
        self,
        file_name: str,
        repo_id: str,
        directory: DirectoryPath = DEFAULT_TMP_DIR,
        hf_access_token: str | None = None,
    ) -> None:
        """Save the model to Hugging Face.

        Args:
            file_name: Name of the file (e.g. "model-something.pt").
            repo_id: ID of the repo to save the model to.
            directory: Directory to save the model to.
            hf_access_token: Hugging Face access token.
        """
        # Save the file
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory / file_name
        self.save(file_path)

        # Upload to Hugging Face
        api = HfApi(token=hf_access_token)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="model",
        )

    @staticmethod
    def load_from_hugging_face(
        file_name: str,
        repo_id: str,
        component_idx: PositiveInt | None = None,
    ) -> "SparseAutoencoder":
        """Load the model from Hugging Face.

        Args:
            file_name: File name of the .pt state file.
            repo_id: ID of the repo to load the model from.
            component_idx: If loading a state dict from a model that has been trained on multiple
                components (e.g. all MLP layers) you may want to to load just one component. In this
                case you can set `component_idx` to the index of the component to load. Note you
                should not set this if you want to load a state dict from a model that has been
                trained on a single component (or if you want to load all components).

        Returns:
            The loaded model.
        """
        local_file = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=file_name,
            revision="main",
        )

        return SparseAutoencoder.load(Path(local_file), component_idx)
