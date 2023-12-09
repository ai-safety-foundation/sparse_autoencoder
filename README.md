# Sparse Autoencoder

[![PyPI](https://img.shields.io/pypi/v/sparse_autoencoder?color=blue)](https://pypi.org/project/transformer-lens/)
![PyPI -
License](https://img.shields.io/pypi/l/sparse_autoencoder?color=blue) [![Checks](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml)
[![Release](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml)

A sparse autoencoder for mechanistic interpretability research.

[![Read the Docs
Here](https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://ai-safety-foundation.github.io/sparse_autoencoder/)](https://ai-safety-foundation.github.io/sparse_autoencoder/)

Train a Sparse Autoencoder [in colab](https://colab.research.google.com/github/ai-safety-foundation/sparse_autoencoder/blob/main/docs/content/demo.ipynb), or install for your project:

```shell
pip install sparse_autoencoder
```

## Features

This library contains:

   1. **A sparse autoencoder model**, along with all the underlying PyTorch components you need to
      customise and/or build your own:
      - Encoder, constrained unit norm decoder and tied bias PyTorch modules in `autoencoder`.
      - L1 and L2 loss modules in `loss`.
      - Adam module with helper method to reset state in `optimizer`.
   2. **Activations data generator** using TransformerLens, with the underlying steps in case you
      want to customise the approach:
      - Activation store options (in-memory or on disk) in `activation_store`.
      - Hook to get the activations from TransformerLens in an efficient way in `source_model`.
      - Source dataset (i.e. prompts to generate these activations) utils in `source_data`, that
        stream data from HuggingFace and pre-process (tokenize & shuffle).
   3. **Activation resampler** to help reduce the number of dead neurons.
   4. **Metrics** that log at various stages of training (e.g. during training, resampling and
      validation), and integrate with wandb.
   5. **Training pipeline** that combines everything together, allowing you to run hyperparameter
      sweeps and view progress on wandb.

## Designed for Research

The library is designed to be modular. By default it takes the approach from [Towards
Monosemanticity: Decomposing Language Models With Dictionary Learning
](https://transformer-circuits.pub/2023/monosemantic-features/index.html), so you can pip install
the library and get started quickly. Then when you need to customise something, you can just extend
the abstract class for that component (e.g. you can extend `AbstractEncoder` if you want to
customise the encoder layer, and then easily drop it in the standard `SparseAutoencoder` model to
keep everything else as is. Every component is fully documented, so it's nice and easy to do this.

## Demo

Check out the demo notebook [docs/content/demo.ipynb](https://github.com/ai-safety-foundation/sparse_autoencoder/blob/main/docs/content/demo.ipynb) for a guide to using this library.

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
