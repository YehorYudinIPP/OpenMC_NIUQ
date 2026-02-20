# OpenMC_NIUQ
A repository for non-intrusive Uncertainty Quantification (NI-UQ) for OpenMC neutronics models.

The package uses [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ) to perform forward
uncertainty propagation (Polynomial Chaos Expansion and quasi-Monte Carlo) for a
fixed-source neutron irradiation model of a **Test Blanket Module (TBM)** built
with [OpenMC](https://openmc.org/).

### Model description

The OpenMC model represents a compact accelerator-driven neutron source with:

* **Rotating Li target wheel** – layered disk (Li → Cu → H₂O cooling → Cu →
  vacuum → graphite shielding → vacuum → Ti window) inside a 50 cm-radius
  ZCylinder.
* **Test Blanket Module (TBM)** – Eurofer-97 RAFM steel casing filled with
  Li₂TiO₃ ceramic either as a solid monoblock or as randomly packed pebbles
  with air between them (selected via `pebbles_or_monoblock` config key).
* **Room** – air-filled 3 m × 2 m × 3 m bounding box with vacuum boundaries.

Run mode is **fixed source**: 14.1 MeV DT-fusion neutrons inside the Li target.

**Quantity of interest (QoI)**: tritium breeding ratio (TBR) – tritium nuclei
produced per source neutron via ⁶Li(n,t)⁴He reactions in the Li₂TiO₃ ceramic.

**Uncertain parameters** for the UQ campaign:

| Parameter | Description | Default | Distribution |
|-----------|-------------|---------|--------------|
| `li_ceramic_density` | Li₂TiO₃ pebble density (g/cm³) | 3.43 | Normal, CoV 2% |
| `li6_enrichment` | Li-6 at% in the ceramic | 7.5 | Normal, CoV 5% |
| `pebble_radius` | pebble radius (cm) | 0.10 | Normal, CoV 5% |
| `packing_fraction` | random packing fraction | 0.30 | Uniform, CoV 5% |

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

