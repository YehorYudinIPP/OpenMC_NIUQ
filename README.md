# OpenMC_NIUQ
A repository for non-intrusive Uncertainty Quantification (NI-UQ) for OpenMC neutronics models.

The package uses [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ) to perform forward
uncertainty propagation (Polynomial Chaos Expansion and quasi-Monte Carlo) for neutronics
models built with [OpenMC](https://openmc.org/).

## Repository structure

```
uq/
├── config/
│   └── model_config.yaml        # YAML model description with UQ parameter specs
├── util/
│   ├── Encoder.py               # Custom YAML encoder for EasyVVUQ
│   ├── utils.py                 # Utility functions (load_config, timestamps, …)
│   └── __init__.py
├── openmc_model_run.py          # Builds and runs an OpenMC model; writes QoIs to CSV
├── easyvvuq_openmc.py           # Main EasyVVUQ UQ script (PCE and QMC)
└── __init__.py
```

## Dependencies

- [OpenMC](https://openmc.org/) (≥ 0.14)
- [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ) (≥ 1.2)
- [chaospy](https://github.com/jonathf/chaospy)
- numpy, pyyaml

Install with conda + pip:

```bash
conda create -n openmc-env -c conda-forge openmc
conda activate openmc-env
pip install easyvvuq chaospy
```

## Usage

Change into the `uq/` directory before running the scripts so that relative
paths in the default config are resolved correctly.

```bash
cd uq
```

### Forward UQ with default settings (PCE, polynomial order 1)

```bash
python easyvvuq_openmc.py
```

### Specify a custom configuration file

```bash
python easyvvuq_openmc.py --config config/model_config.yaml
```

### Choose the UQ scheme and its parameters

```bash
# Polynomial Chaos Expansion, order 2
python easyvvuq_openmc.py --uq-scheme pce --p-order 2

# Quasi-Monte Carlo, 256 samples
python easyvvuq_openmc.py --uq-scheme qmc --n-samples 256
```

### Run a single model evaluation (for testing)

```bash
python openmc_model_run.py --config config/model_config.yaml
```

## Configuration

The YAML configuration file (`uq/config/model_config.yaml`) describes the OpenMC
model and the uncertainty specifications for each parameter.  Uncertain parameters
include a `mean`, `relative_stdev`, and `pdf` (probability density function) field:

```yaml
materials:
  fuel:
    enrichment:
      mean: 3.1          # nominal wt% U-235
      relative_stdev: 0.05
      pdf: normal        # 'normal' or 'uniform'
```

Fixed parameters (no uncertainty) are plain scalar values.

