# Dataloaders

This directory contains dataloaders for sequence modeling tasks used in LoongSpike experiments.

## Supported Datasets

### Auto-Download Datasets
- **MNIST**: Sequential MNIST for digit classification
- **Speech Commands**: Audio classification dataset
- **CIFAR-10**: Sequential CIFAR-10 (grayscale, for LRA Image task)
- **IMDB**: Character-level sentiment classification (for LRA Text task)

### Manual Download (LRA Benchmark)
- **ListOps**: Hierarchical structure prediction
- **PathFinder**: Long-range spatial dependency
- **AAN**: Document retrieval task

## Data Path

By default, data is downloaded to `./data/`. You can customize this:

**Environment Variable:**
```bash
export DATA_PATH=/path/to/data
```

**Config Override:**
```bash
python -m train experiment=spikingssm/cifar +dataset.data_dir=/path/to/data
```

## LRA Manual Download

For ListOps, PathFinder, and AAN tasks, download the LRA benchmark datasets.

**Note**: The original LRA repository may have limited access. You can find the datasets from:
- [LRA GitHub](https://github.com/google-research/long-range-arena) (original source)
- Alternative sources: Search for "Long Range Arena benchmark datasets" or check related papers' repositories

**Setup after download:**

```bash
cd data
# After obtaining lra_release.gz
tar xvf lra_release.gz
mv lra_release/lra_release/listops-1000 listops
mv lra_release/lra_release/tsv_data aan
mkdir pathfinder
mv lra_release/lra_release/pathfinder* pathfinder/
rm -r lra_release
```

**Expected structure:**
```
./data/
  mnist/              (auto-download)
  SpeechCommands/     (auto-download)
  cifar/              (auto-download)
  imdb/               (auto-download)
  listops/            (manual download)
  aan/                (manual download)
  pathfinder/         (manual download)
```

## Dataset Files

- **base.py**: Base `SequenceDataset` class
- **basic.py**: MNIST dataset
- **vision.py**: CIFAR-10 dataset
- **audio.py**: Speech Commands dataset
- **lra.py**: LRA benchmark datasets (ListOps, PathFinder, AAN, IMDB)

## Usage

```bash
# MNIST
python -m train experiment=spikingssm/mnist

# Speech Commands
python -m train experiment=spikingssm/sc

# LRA - Image (CIFAR-10)
python -m train experiment=spikingssm/cifar

# LRA - Text (IMDB)
python -m train experiment=spikingssm/imdb

# LRA - ListOps
python -m train experiment=spikingssm/listops

# LRA - PathFinder
python -m train experiment=spikingssm/pathfinder

# LRA - Retrieval (AAN)
python -m train experiment=spikingssm/aan
```

