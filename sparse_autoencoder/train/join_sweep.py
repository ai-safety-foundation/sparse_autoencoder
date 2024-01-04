"""Join an existing Weights and Biases sweep, as a new agent."""
import argparse

from sparse_autoencoder.train.sweep import sweep


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Join an existing W&B sweep.")
    parser.add_argument(
        "--id", type=str, default=None, help="Sweep ID for the existing sweep.", required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    sweep(sweep_id=args.id)
