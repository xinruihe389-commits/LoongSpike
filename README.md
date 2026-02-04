# LoongSpike: Fractional-Order Spiking State Space Models

Efficient long sequence modeling with fractional-order spiking neural networks.

---

## Overview

**LoongSpike** is a spiking sequence modeling framework that integrates fractional-order state space models (f-SSM) into spiking neural networks, enabling efficient long-range dependency modeling with sparse computation.

**Key Features:**
- **Fractional-Order Dynamics**: Replaces standard first-order Markovian transitions with fractional-order dynamics to capture long-memory effects and alleviate the memoryless bottleneck of traditional SNNs
- **Parallelizable Computation**: Reformulates fractional operators into a parallelizable state-space representation, enabling efficient training on long sequences without sequential update constraints
- **Energy Efficiency**: Maintains sparse spiking computation while achieving superior accuracy on long-sequence benchmarks
- **Strong Performance**: Consistently outperforms state-of-the-art SNNs on Sequential MNIST, Long Range Arena (LRA), and Speech Commands tasks

---

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+ 
- loguru

### Setup

```bash
# Create environment
conda create -n loongspike python=3.8
conda activate loongspike

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Training

**Train LoongSpike (fractional-order):**
```bash
# CIFAR-10 (LRA Image task)
python -m train experiment=spikingssm/cifar

# Speech Commands
python -m train experiment=spikingssm/sc

# LRA ListOps
python -m train experiment=spikingssm/listops
```

**Train standard SpikingSSM (baseline):**
```bash
python -m train experiment=spikingssm/cifar model._name_=spikingssm
```

### Evaluation

```bash
# Evaluate a trained model
python checkpoints/evaluate.py \
  --checkpoint outputs/path/to/checkpoint.ckpt \
  --config configs/experiment/spikingssm/cifar.yaml
```

### Configuration

Override hyperparameters via command line:
```bash
python -m train \
  experiment=spikingssm/cifar \
  model.d_model=512 \
  model.n_layers=6 \
  trainer.max_epochs=200 \
  optimizer.lr=0.001
```

---

## Model Architecture

### Two Versions

1. **Standard SpikingSSM** (`ss4d.py`): Integer-order baseline
2. **LoongSpike** (`loongspike.py`): Fractional-order (our contribution)

### Key Components

**Fractional-order dynamics:**
```
D^α h(t) = Ah(t) + Bx(t)
```
where `D^α` is the Caputo fractional derivative, `α ∈ (0, 1]`.

**Exponential approximation:**
```
K_α(t) ≈ Σ w_m exp(-λ_m t)
```

---

## Experiments

### Supported Tasks

| Task | Dataset | Type | Auto-download |
|------|---------|------|---------------|
| LRA Image | Sequential CIFAR-10 | Image | ✓ |
| LRA Text | IMDB | Text | ✓ |
| LRA ListOps | ListOps | Reasoning | Manual |
| LRA PathFinder | PathFinder | Vision | Manual |
| LRA Retrieval | AAN | Text | Manual |
| Audio | Speech Commands | Audio | ✓ |
| Vision | MNIST | Image | ✓ |

---

## Acknowledgments

This codebase builds upon and references implementations from:

- **[S4 (Structured State Spaces)](https://github.com/state-spaces/s4)** (Apache License 2.0): State space model architecture, kernel computation, and training infrastructure
- **[SpikingSSMs](https://github.com/shenshuaijie/SDN)** (MIT License): Spiking neural network integration with state space models

We thank the authors for open-sourcing their code, which provided the foundation for our fractional-order extensions.

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

Portions of this code are derived from:
- **S4** (Apache License 2.0) - See [LICENSE-APACHE](LICENSE-APACHE)
- **SpikingSSMs** (MIT License) - See [LICENSE-MIT](LICENSE-MIT)

