# Model Configurations

The `model/` configs define model architectures and follow the structure of the `src/models/` code folder.

## Structure

Model configs consist of two main components:

### Backbones
Top-level configs define model backbones that are composed of repeatable blocks of core layers.
- `base.yaml` - Simple isotropic residual backbone (in the style of ResNets and Transformers)

### Layers
Layer configs are defined in `layer/`. Each instantiates a `src.models.sequence.base.SequenceModule` which maps an input sequence to an output sequence.

Available layers:
- `layer/s4.yaml` - S4 (Structured State Space) layer
- `layer/s4d.yaml` - S4D (diagonal S4) layer, used in SpikingSSM
- `layer/s4d_example.yaml` - Example S4D configuration

## Models

Full model configs that combine a backbone with a choice of inner layer:

- `s4.yaml` - Basic isotropic S4 model

## Usage

To use a model config:

```bash
# Use the S4 model
python -m train model=s4

# Use a backbone with custom layer
python -m train model=base model.layer=s4d

# Override model parameters
python -m train model=s4 model.d_model=512 model.n_layers=6
```

## Creating Custom Models

Models can be customized by:
1. Choosing a backbone (e.g., `base.yaml`)
2. Selecting a layer type (e.g., `layer/s4d.yaml`)
3. Overriding parameters via command line or experiment configs

See `configs/experiment/` for complete examples.

