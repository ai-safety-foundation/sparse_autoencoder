# Sparse Autoencoder

[![PyPI](https://img.shields.io/pypi/v/sparse_autoencoder?color=blue)](https://pypi.org/project/transformer-lens/)
![PyPI - License](https://img.shields.io/pypi/l/sparse_autoencoder?color=blue)
[![Checks](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/checks.yml)
[![Release](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml/badge.svg)](https://github.com/alan-cooney/sparse_autoencoder/actions/workflows/release.yml)

A sparse autoencoder for mechanistic interpretability research.

```shell
pip install sparse_autoencoder
```

## Quick Start

Check out the [demo notebook](demo) for a guide to using this library.

We also highly recommend skimming the reference docs to see all the features that are available.

## Features

This library contains:

   1. **A sparse autoencoder model**, along with all the underlying PyTorch components you need to
      customise and/or build your own:
      - Encoder, constrained unit norm decoder and tied bias PyTorch modules in
        [sparse_autoencoder.autoencoder][].
      - L1 and L2 loss modules in [sparse_autoencoder.loss][].
      - Adam module with helper method to reset state in [sparse_autoencoder.optimizer][].
   2. **Activations data generator** using TransformerLens, with the underlying steps in case you
      want to customise the approach:
      - Activation store options (in-memory or on disk) in [sparse_autoencoder.activation_store][].
      - Hook to get the activations from TransformerLens in an efficient way in
        [sparse_autoencoder.source_model][].
      - Source dataset (i.e. prompts to generate these activations) utils in
        [sparse_autoencoder.source_data][], that stream data from HuggingFace and pre-process
        (tokenize & shuffle).
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
the abstract class for that component (every component is documented so that it's easy to do this).
