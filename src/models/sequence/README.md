# Sequence Models

This folder contains the modular sequence modeling framework that underlies SpikingSSM and LoongSpike.

## Directory Structure

```
base.py      Base SequenceModule interface
backbones/   Model architectures and building blocks
kernels/     SSM kernel implementations (core of S4 and SpikingSSM)
modules/     Reusable sequence modules (S4 blocks, FFN, pooling)
```

## Core Components

### SequenceModule Interface

The `SequenceModule` class ([base.py](base.py)) defines the standard interface for all sequence models:
- Input: `(batch_size, sequence_length, d_input)`
- Output: `(batch_size, sequence_length, d_output)`

All models in this codebase implement this interface for consistency.

### SSM Kernels

The SSM implementations are in [kernels/](kernels/):
- **ssm.py**: Core State Space Model implementation (used by SpikingSSM)
- **dplr.py**: DPLR parameterization for efficient computation
- **fftconv.py**: FFT-based convolution for parallel training
- **kernel.py**: Generic kernel functions

### Model Backbones

The [backbones/](backbones/) directory contains:
- **model.py**: Main model architecture with residual connections and normalization
- **block.py**: Basic building blocks for composing models

### Sequence Modules

The [modules/](modules/) directory provides reusable components:
- **s4block.py**: S4 block implementation
- **ffn.py**: Feed-forward network
- **pool.py**: Pooling operations

## Usage

SpikingSSM models are built on top of this framework. See the main [models README](../README.md) for usage examples.
