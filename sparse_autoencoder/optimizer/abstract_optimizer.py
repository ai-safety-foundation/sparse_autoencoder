"""Abstract optimizer with reset."""
from abc import ABC, abstractmethod


class AbstractOptimizerWithReset(ABC):
    """Abstract optimizer with reset."""

    @abstractmethod
    def reset_state_all_parameters(self) -> None:
        """Reset the state for all parameters.

        Resets any optimizer state (e.g. momentum). This is for use after manually editing model
            parameters (e.g. with activation resampling).
        """
        raise NotImplementedError
