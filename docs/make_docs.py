"""Build the API Documentation."""
from pathlib import Path
import shutil
import subprocess
from typing import Any


# Docs Directories
CURRENT_DIR = Path(__file__).parent
SOURCE_PATH = CURRENT_DIR / "../docs/source"
BUILD_PATH = CURRENT_DIR / "../docs/build"
PACKAGE_DIR = CURRENT_DIR.parent
DEMOS_DIR = CURRENT_DIR.parent
GENERATED_DIR = CURRENT_DIR.parent / "docs/source/generated"


def copy_demos(_app: Any | None = None) -> None:
    """Copy demo notebooks to the generated directory."""
    if not GENERATED_DIR.exists():
        GENERATED_DIR.mkdir()

    copy_to_dir = GENERATED_DIR / "demos"
    notebooks_to_copy = [
        "benchmarks.ipynb",
        "demo.ipynb",
    ]

    if copy_to_dir.exists():
        shutil.rmtree(copy_to_dir)

    copy_to_dir.mkdir()
    for filename in notebooks_to_copy:
        shutil.copy(DEMOS_DIR / filename, copy_to_dir)


def build_docs() -> None:
    """Build the docs."""
    copy_demos()

    # Generating docs
    subprocess.run(
        [
            "sphinx-build",
            SOURCE_PATH,
            BUILD_PATH,
            # Nitpicky mode (warn about all missing references)
            # "-n",  # noqa: ERA001
            # Turn warnings into errors
            "-W",
        ],
        check=True,
    )


def docs_hot_reload() -> None:
    """Hot reload the docs."""
    copy_demos()

    subprocess.run(
        [
            "sphinx-autobuild",
            "--watch",
            str(PACKAGE_DIR) + "," + str(DEMOS_DIR),
            "--open-browser",
            SOURCE_PATH,
            BUILD_PATH,
        ],
        check=True,
    )
