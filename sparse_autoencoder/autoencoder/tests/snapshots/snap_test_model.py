"""Auto-Generated Snapshot tests."""

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots["test_representation Model string representation"] = """SparseAutoencoder(
  (encoder): Sequential(
    (0): TiedBias(position=pre_encoder)
    (1): Linear(in_features=3, out_features=6, bias=True)
    (2): ReLU()
  )
  (decoder): Sequential(
    (0): ConstrainedUnitNormLinear(in_features=6, out_features=3, bias=False)
    (1): TiedBias(position=post_decoder)
  )
)"""
