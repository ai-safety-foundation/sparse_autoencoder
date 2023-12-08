"""Test wandb sweep types."""
from dataclasses import dataclass, field

from sparse_autoencoder.train.utils.wandb_sweep_types import (
    Method,
    Metric,
    NestedParameter,
    Parameter,
    Parameters,
    WandbSweepConfig,
)


class TestNestedParameter:
    """NestedParameter tests."""

    def test_to_dict(self) -> None:
        """Test to_dict method."""

        @dataclass(frozen=True)
        class DummyNestedParameter(NestedParameter):
            nested_property: Parameter[float] = field(default=Parameter(1.0))

        dummy = DummyNestedParameter()

        # It should be in the nested "parameters" key.
        assert dummy.to_dict() == {"parameters": {"nested_property": {"value": 1.0}}}


class TestWandbSweepConfig:
    """WandbSweepConfig tests."""

    def test_to_dict(self) -> None:
        """Test to_dict method."""

        @dataclass(frozen=True)
        class DummyNestedParameter(NestedParameter):
            nested_property: Parameter[float] = field(default=Parameter(1.0))

        @dataclass
        class DummyParameters(Parameters):
            nested: DummyNestedParameter = field(default=DummyNestedParameter())
            top_level: Parameter[float] = field(default=Parameter(1.0))

        dummy = WandbSweepConfig(
            parameters=DummyParameters(), method=Method.GRID, metric=Metric(name="total_loss")
        )

        assert dummy.to_dict() == {
            "method": "grid",
            "metric": {"goal": "minimize", "name": "total_loss"},
            "parameters": {
                "nested": {
                    "parameters": {"nested_property": {"value": 1.0}},
                },
                "top_level": {"value": 1.0},
            },
        }
