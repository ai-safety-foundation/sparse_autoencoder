"""Generate the code reference pages."""


from pathlib import Path

import mkdocs_gen_files


def is_source_file(file: Path) -> bool:
    """Checks an individual file to see if they are source files for Sparse Encoder."""
    return "test" not in str(file)


nav = mkdocs_gen_files.Nav()  # type: ignore

CURRENT_DIR = Path(__file__).parent
REPO_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = REPO_ROOT / "sparse_autoencoder"
REFERENCE_DIR = CURRENT_DIR / "content/reference"

python_files = Path(PROJECT_ROOT).rglob("*.py")
source_files = filter(is_source_file, python_files)

for path in sorted(source_files):
    module_path = path.relative_to(PROJECT_ROOT).with_suffix("")  #
    doc_path = path.relative_to(PROJECT_ROOT).with_suffix(".md")  #
    full_doc_path = Path(REFERENCE_DIR, doc_path)  #

    parts = list(module_path.parts)

    if parts[-1] == "__init__":  #
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    slug = ["Home"] if not len(parts) else parts

    nav[slug] = doc_path.as_posix()  # type: ignore

    parts.insert(0, "sparse_autoencoder")

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  #
        identifier = ".".join(parts)  #
        fd.write(f"::: {identifier}")  #

    mkdocs_gen_files.set_edit_path(full_doc_path, path)  #


with mkdocs_gen_files.open(REFERENCE_DIR / "SUMMARY.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())
