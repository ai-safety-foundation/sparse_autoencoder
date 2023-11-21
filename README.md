# Sparse Autoencoder

[![PyPI](https://img.shields.io/pypi/v/sparse_autoencoder?color=blue)](https://pypi.org/project/transformer-lens/)
![PyPI -
License](https://img.shields.io/pypi/l/sparse_autoencoder?color=blue) [![Checks](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml)
[![Release](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml)

A sparse autoencoder for mechanistic interpretability research.

```shell
pip install sparse_autoencoder
```

## Demo

Check out the demo notebook for a guide to using this library.

## Contributing

This project uses [Poetry](https://python-poetry.org) for dependency management, and
[PoeThePoet](https://poethepoet.natn.io/installation.html) for scripts. After checking out the repo,
we recommend setting poetry's config to create the `.venv` in the root directory (note this is a
global setting) and then installing with the dev and demos dependencies.

```shell
poetry config virtualenvs.in-project true
poetry install --with dev,demos
```

### Checks

For a full list of available commands (e.g. `test` or `typecheck`), run this in your terminal
(assumes the venv is active already).

```shell
poe
```
