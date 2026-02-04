# Models

This directory provides a modular implementation of sequence models, with a focus on State Space Models (SSMs) and the SpikingSSM architecture.

## Directory Structure

```
spike/       SpikingSSM models (ss4d.py: baseline, loongspike.py: our contribution)
sequence/    Modular sequence model framework
functional/  Mathematical utilities for SSMs
hippo/       HiPPO operator utilities
nn/          Neural network components (activation, normalization, etc.)
baselines/   Baseline models for comparison (optional)
```

## Core Models

### SpikingSSM (spike/)

This repository contains two versions of SpikingSSM models:

1. **Standard SpikingSSM** (`ss4d.py`): The baseline SpikingSSM model with integer-order dynamics
2. **LoongSpike** (`loongspike.py`): Our main contribution - Fractional-order SpikingSSM with enhanced memory capabilities

**Supporting modules:**
- **neuron.py**: Spiking neuron models
- **surrogate.py**: Surrogate gradient functions for training spiking neurons

### Sequence Models (sequence/)

A modular framework for building sequence models. Key components:

- **kernels/**: SSM kernel implementations
  - `ssm.py`: Core State Space Model implementation
  - `kernel.py`: Kernel functions
  - `dplr.py`: DPLR parameterization
  - `fftconv.py`: FFT-based convolution

- **modules/**: Reusable sequence modules
  - `s4block.py`: S4 block implementation
  - `ffn.py`: Feed-forward network
  - `pool.py`: Pooling operations

- **backbones/**: Model architectures
  - `model.py`: Main model backbone
  - `block.py`: Basic building blocks

- **base.py**: Base `SequenceModule` interface

See [sequence/README.md](sequence/README.md) for detailed documentation.

## Mathematical Utilities

### HiPPO (hippo/)

HiPPO is the mathematical framework underlying S4 and related models:
- **hippo.py**: HiPPO matrix definitions
- **transition.py**: State transition matrices

### Functional (functional/)

Mathematical utilities for SSM operations:
- **cauchy.py**: Cauchy kernel computations
- **vandermonde.py**: Vandermonde matrix operations
- **krylov.py**: Krylov methods
- **toeplitz.py**: Toeplitz matrix operations

## Neural Network Components (nn/)

Reusable neural network modules:
- **activation.py**: Activation functions
- **normalization.py**: Normalization layers
- **dropout.py**: Dropout variants
- **linear.py**: Linear layers
- **initialization.py**: Weight initialization
- **residual.py**: Residual connections

## Baseline Models (baselines/)

Optional baseline models for comparison:
- **transformer.py**: Transformer baseline
- **lstm.py**: LSTM baseline
- **gru.py**: GRU baseline
- **resnet.py**: ResNet baseline

These can be used for comparison experiments:
```bash
python -m train experiment=spikingssm/cifar model=transformer
python -m train experiment=spikingssm/cifar model=lstm
```

## Usage

### Using SpikingSSM Models

```bash
# Train with standard SpikingSSM (baseline, integer-order)
python -m train experiment=spikingssm/cifar model._name_=spikingssm

# Train with LoongSpike (our contribution, fractional-order)
python -m train experiment=spikingssm/cifar model._name_=loongspikingssm

# Or simply use the default (LoongSpike)
python -m train experiment=spikingssm/cifar
```

### Using Baseline Models

```bash
# Compare with Transformer
python -m train experiment=spikingssm/cifar model=transformer

# Compare with LSTM
python -m train experiment=spikingssm/cifar model=lstm
```

## Modular Design

The `SequenceModule` interface in `sequence/base.py` provides a unified API for all sequence models. This allows:

1. **Flexible composition**: Stack different sequence modules
2. **Easy experimentation**: Swap models without changing other code
3. **Consistent interface**: All models follow the same API

Example of the interface:
```python
class SequenceModule(nn.Module):
    def forward(self, x, state=None, **kwargs):
        """
        x: (batch, length, d_input)
        Returns: (batch, length, d_output), state
        """
        pass
```

## Adding Custom Models

To add a new sequence model:

1. Inherit from `SequenceModule` in `sequence/base.py`
2. Implement the `forward` method
3. Register the model in `src/utils/registry.py`
4. Create a config file in `configs/model/`

See existing models in `sequence/modules/` for examples.

## Model Comparison

| Model | Type | Key Features |
|-------|------|--------------|
| **LoongSpike** | Fractional-order SpikingSSM | Enhanced long-term memory, fractional dynamics |
| **SpikingSSM** | Integer-order SpikingSSM | Standard spiking dynamics, baseline model |
| Transformer | Attention-based | Self-attention mechanism |
| LSTM | RNN-based | Gated recurrent units |

