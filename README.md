# Sparse Autoencoder

[![Pypi](https://img.shields.io/pypi/v/sparse_autoencoder?color=blue)](https://pypi.org/project/transformer-lens/)
![PyPI -
License](https://img.shields.io/pypi/l/sparse_autoencoder?color=blue) [![Checks](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml)
[![Release](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml)

A sparse autoencoder for mechanistic interpretability research.

**Note**: This package is in alpha and likely to have breaking changes regularly.

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
