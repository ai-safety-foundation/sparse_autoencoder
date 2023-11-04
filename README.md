# Sparse Autoencoder

[![Checks](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml)
[![Release](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml)

**Note**: This package is in alpha and likely to have breaking changes regularly.

A sparse autoencoder for mechanistic interpretability research.

```shell
pip install sparse_autoencoder
```

## Demo

Check out the demo notebook for a guide to using this library.

## Contributing

This project uses [Poetry](https://python-poetry.org) for dependency management. After checking out
the repo, install all dependencies with:

```shell
poetry install --with dev,demos
```

Then to run all checks locally:

```shell
poetry run ruff check sparse_autoencoder
poetry run pyright
poetry run pytest
```
