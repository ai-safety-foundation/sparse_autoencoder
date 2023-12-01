"""Auto-generate reference documentation based on docstrings."""
from pathlib import Path
import shutil

import mkdocs_gen_files


CURRENT_DIR = Path(__file__).parent
REPO_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = REPO_ROOT / "sparse_autoencoder"
REFERENCE_DIR = CURRENT_DIR / "content/reference"


def reset_reference_dir() -> None:
    """Reset the reference directory to its initial state."""
    # Unlink the directory including all files
    shutil.rmtree(REFERENCE_DIR, ignore_errors=True)
    REFERENCE_DIR.mkdir(parents=True)


def is_source_file(file: Path) -> bool:
    """Check if the provided file is a source file for Sparse Encoder.

    Args:
        file: The file path to check.

    Returns:
        bool: True if the file is a source file, False otherwise.
    """
    return "test" not in str(file)


def process_path(path: Path) -> tuple[Path, Path, Path]:
    """Process the given path for documentation generation.

    Args:
        path: The file path to process.

    Returns:
        A tuple containing module path, documentation path, and full documentation path.
    """
    module_path = path.relative_to(PROJECT_ROOT).with_suffix("")
    doc_path = path.relative_to(PROJECT_ROOT).with_suffix(".md")
    full_doc_path = Path(REFERENCE_DIR, doc_path)

    if module_path.name == "__init__":
        module_path = module_path.parent
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    return module_path, doc_path, full_doc_path


def generate_documentation(path: Path, module_path: Path, full_doc_path: Path) -> None:
    """Generate documentation for the given source file.

    Args:
        path: The source file path.
        module_path: The module path.
        full_doc_path: The full documentation file path.
    """
    if module_path.name == "__main__":
        return

    # Get the mkdocstrings identifier for the module
    parts = list(module_path.parts)
    parts.insert(0, "sparse_autoencoder")
    identifier = ".".join(parts)

    # Read the first line of the file docstring, and set as the header
    with path.open() as fd:
        first_line = fd.readline()
        first_line_without_docstring = first_line.replace('"""', "").strip()
        first_line_without_last_dot = first_line_without_docstring.rstrip(".")
        title = first_line_without_last_dot or module_path.name

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"# {title}" + "\n\n" + f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)


def generate_nav_file(nav: mkdocs_gen_files.nav.Nav, reference_dir: Path) -> None:
    """Generate the navigation file for the documentation.

    Args:
        nav: The navigation object.
        reference_dir: The directory to write the navigation file.
    """
    with mkdocs_gen_files.open(reference_dir / "SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


def run() -> None:
    """Handle the generation of reference documentation for Sparse Encoder."""
    reset_reference_dir()
    nav = mkdocs_gen_files.Nav()  # type: ignore

    python_files = PROJECT_ROOT.rglob("*.py")
    source_files = filter(is_source_file, python_files)

    for path in sorted(source_files):
        module_path, doc_path, full_doc_path = process_path(path)
        generate_documentation(path, module_path, full_doc_path)

        url_slug_parts = list(module_path.parts)

        # Don't create a page for the main __init__.py file (as this includes most exports).
        if not url_slug_parts:
            continue

        nav[url_slug_parts] = doc_path.as_posix()  # type: ignore

    generate_nav_file(nav, REFERENCE_DIR)


run()
