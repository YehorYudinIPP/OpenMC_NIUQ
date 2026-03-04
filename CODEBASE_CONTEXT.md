# Codebase Context – OpenMC_NIUQ

> **Purpose of this file**: Provide a quick-reference context document so that AI
> coding agents (and new contributors) can understand the repository structure,
> conventions, and key design decisions without re-exploring the entire codebase
> each session.
>
> **Last updated**: 2026-03-04

---

## 1. Project Overview

**OpenMC_NIUQ** is a Python framework for **Non-Intrusive Uncertainty
Quantification (NI-UQ)** applied to [OpenMC](https://openmc.org/) neutronics
simulations.  It models a **Test Blanket Module (TBM)** — an accelerator-driven
neutron source with a Li₂TiO₃ ceramic pebble-bed — and propagates parameter
uncertainties through the simulation using
[EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ).

**Key capabilities:**
- Polynomial Chaos Expansion (PCE) and Quasi-Monte Carlo (QMC) uncertainty propagation
- Auto-discovery of uncertain parameters from YAML configuration
- Sobol sensitivity analysis (first-order, total, second-order interactions)
- Automated visualization of QoI statistics, distributions, and sensitivity indices

**License:** CC0 1.0 Universal (Public Domain)

---

## 2. Directory Structure

```
OpenMC_NIUQ/
├── CODEBASE_CONTEXT.md              # ← This file (agent/contributor context)
├── README.md                        # User-facing documentation
├── LICENSE                          # CC0 1.0 Universal
├── requirements.txt                 # Python dependencies (pip)
├── .gitignore                       # Excludes __pycache__, OpenMC outputs, pickles, logs, plots
│
├── uq/                              # Main Python package
│   ├── __init__.py                  # Exports: load_config, add_timestamp_to_filename,
│   │                                #   get_openmc_python, validate_execution_setup,
│   │                                #   save_sa_results_yaml
│   ├── config/
│   │   ├── model_config.yaml            # Primary TBM pebble-bed configuration
│   │   └── model_config_graphite_liner.yaml  # Variant with 0.5 cm graphite liner
│   ├── util/
│   │   ├── __init__.py              # Exports: YAMLEncoder, AdvancedYAMLEncoder,
│   │   │                            #   load_config, add_timestamp_to_filename, etc.
│   │   ├── Encoder.py               # YAMLEncoder (simple) + AdvancedYAMLEncoder (nested paths)
│   │   └── utils.py                 # load_config, add_timestamp_to_filename,
│   │                                #   get_openmc_python, validate_execution_setup,
│   │                                #   save_sa_results_yaml
│   ├── openmc_model_run.py          # Builds OpenMC TBM model from YAML; runs single sim; writes CSV
│   ├── openmc_model_run_graphite_liner.py  # Graphite-liner model variant
│   ├── easyvvuq_openmc.py           # Main EasyVVUQ UQ campaign orchestration script
│   ├── visualisation.py             # Post-campaign plotting utilities (8+ plot types)
│   └── visualise_uq_results.py      # Standalone re-visualization from saved pickle results
│
└── tests/
    ├── __init__.py                  # Empty
    ├── test_config_and_discovery.py # Config loading, _get_mean(), auto-discovery, logging
    ├── test_graphite_liner_model.py # Graphite-liner variant tests
    └── test_visualisation.py        # Visualization tests with mock EasyVVUQ results
```

---

## 3. Tech Stack & Dependencies

| Dependency       | Version   | Install method | Purpose                              |
|------------------|-----------|----------------|--------------------------------------|
| `openmc`         | ≥ 0.14    | conda-forge    | Neutron transport simulation         |
| `easyvvuq`       | ≥ 1.2     | pip            | UQ campaign orchestration            |
| `chaospy`        | ≥ 4.3     | pip            | Statistical distributions & PCE      |
| `numpy`          | any       | pip            | Numerical computing                  |
| `pyyaml`         | any       | pip            | YAML configuration parsing           |
| `matplotlib`     | any       | pip            | Visualization / plotting             |
| `pytest`         | any       | pip (dev)      | Testing framework                    |

**Installation:**
```bash
conda create -n openmc-env -c conda-forge openmc
conda activate openmc-env
pip install -r requirements.txt
```

---

## 4. How to Build, Test, and Run

There is no compilation step (pure Python). No linting or formatting tools are
configured.

### Testing
```bash
# Run all tests (from repo root):
pytest

# Run a specific test file:
pytest tests/test_config_and_discovery.py
pytest tests/test_visualisation.py
pytest tests/test_graphite_liner_model.py
```

Tests do **not** require OpenMC or EasyVVUQ to be installed — they exercise
pure-Python configuration logic and use mocking for visualization.

### Running the project
```bash
cd uq/

# Single model evaluation (requires OpenMC):
python openmc_model_run.py --config config/model_config.yaml

# UQ campaign with defaults (PCE, polynomial order 1):
python easyvvuq_openmc.py

# Custom UQ scheme:
python easyvvuq_openmc.py --uq-scheme pce --p-order 2
python easyvvuq_openmc.py --uq-scheme qmc --n-samples 256
```

---

## 5. Architecture & Data Flow

```
model_config.yaml
       │
       ▼
easyvvuq_openmc.py ─── discover_uncertain_parameters()
       │                     │
       │                     ▼
       │               define_model_parameters()
       │               define_parameter_distributions()
       │                     │
       │                     ▼
       │               prepare_uq_campaign()
       │                 ├── AdvancedYAMLEncoder (writes sampled YAML configs)
       │                 ├── PCESampler or QMCSampler
       │                 └── ExecuteLocal → openmc_model_run.py --config <sampled>.yaml
       │                     │
       │                     ▼
       │               run_uq_campaign()
       │                     │
       │                     ▼
       │               analyse_uq_results()
       │                 ├── Sobol indices (1st-order, total, 2nd-order)
       │                 └── Descriptive statistics (mean, std, percentiles)
       │                     │
       │                     ▼
       └──────────────► visualisation.py
                          ├── QoI statistics bar charts
                          ├── Sobol index bar charts & heatmaps
                          ├── Distribution histograms
                          ├── Input uncertainty PDFs
                          └── Saved to plots_openmc_uq_<timestamp>/
```

---

## 6. Key Conventions & Patterns

### Configuration system
- **YAML-driven**: All model and UQ parameters live in `uq/config/model_config.yaml`.
- **Uncertain parameters** are expressed as dicts with `mean`, `relative_stdev`, `pdf` keys.
- **Fixed parameters** are plain scalar values.
- **Auto-discovery**: `discover_uncertain_parameters()` recursively walks the config
  to find all dicts containing a `mean` key.

### Parameter naming
- Discovered parameter names are formed from the **last two path segments** joined
  with `_` (e.g., `materials.li_ceramic.density` → `li_ceramic_density`).

### The `_get_mean()` helper
- Defined in `openmc_model_run.py` (and its graphite-liner variant).
- Returns `value` if scalar, `value["mean"]` if dict, or `0.0` as fallback.
- **All uncertain and fixed parameters must be read through `_get_mean()`** in the
  model builder so that both formats work transparently.

### Adding a new uncertain parameter (3 steps)
1. Add the parameter in `model_config.yaml` with `mean`, `relative_stdev`, `pdf` keys.
2. Read it in `openmc_model_run.py` via `_get_mean()`.
3. **Done** — `easyvvuq_openmc.py` auto-discovers it (no code changes needed there).

### Logging
- Logger name: `"openmc_uq"` (via `logging.getLogger("openmc_uq")`).
- Dual output: console (INFO+) and timestamped log file (DEBUG+).
- Log file: `openmc_uq_campaign_YYYYMMDD_HHMMSS.log`.

### Result persistence
- Campaign config pickle: `uq_campaign_config_<timestamp>.pickle`
- Analysis results pickle: `analysis_results_openmc_uq_<timestamp>.pickle`
- Plots directory: `plots_openmc_uq_<timestamp>/`

### Visualization labels
- Human-readable labels defined in `visualisation.py` as `_QOI_LABELS` and `_PARAM_LABELS`.
- Support LaTeX notation (e.g., `$^6$Li`, `$_2$TiO$_3$`).

### Encoding (EasyVVUQ integration)
- `YAMLEncoder`: Simple `$param_name$` placeholder substitution.
- `AdvancedYAMLEncoder`: Nested dot-notation path support (e.g., `geometry.pebble_radius.mean`).
- Both support parameter type conversion and fixed parameter injection.

---

## 7. Quantities of Interest (QoIs)

| QoI key                   | Description                                           |
|---------------------------|-------------------------------------------------------|
| `tritium_production_rate` | Tritium breeding ratio (TBR): ⁶Li(n,t)⁴He per source neutron |
| `total_neutron_flux`      | Total flux intensity across all materials              |
| `tbm_incident_flux`       | Neutron current incident on TBM casing surfaces        |
| `tbm_inner_flux`          | Neutron flux inside the TBM (Li₂TiO₃ ceramic)         |
| `tbm_heating`             | Nuclear heating in TBM ceramic (eV/source neutron)     |
| `tbm_neutron_leakage`     | Neutron leakage current from TBM casing surfaces       |

---

## 8. Uncertain Parameters (current defaults)

| Parameter            | Default | Distribution | CoV  | Config path                          |
|----------------------|---------|--------------|------|--------------------------------------|
| `li_ceramic_density` | 3.43    | Normal       | 2%   | `materials.li_ceramic.density`       |
| `li6_enrichment`     | 7.5     | Normal       | 5%   | `materials.li_ceramic.li6_enrichment`|
| `pebble_radius`      | 0.10    | Normal       | 5%   | `geometry.pebble_radius`             |
| `packing_fraction`   | 0.30    | Uniform*     | 5%   | `geometry.packing_fraction`          |
| `graphite_thickness` | 0.70    | Normal       | 5%   | `geometry.graphite_thickness`        |

\* `packing_fraction` is currently a plain scalar (commented out as uncertain in the config).

---

## 9. Test Structure

All tests are in the `tests/` directory and use **pytest**.  Tests do **not**
require OpenMC — they use pure Python logic and mocking.

| Test file                         | What it covers                                   |
|-----------------------------------|--------------------------------------------------|
| `test_config_and_discovery.py`    | `_get_mean()`, parameter auto-discovery, logging, default QoIs, distribution building |
| `test_graphite_liner_model.py`    | Graphite-liner config loading, `_get_mean()` in variant, parameter presence |
| `test_visualisation.py`           | Plot functions with mock EasyVVUQ results objects |

### Test import pattern
Tests add `uq/` to `sys.path` so they can import directly from `openmc_model_run`
and `easyvvuq_openmc` (see path manipulation at top of test files).

---

## 10. CI/CD & Tooling

- **No CI/CD pipeline** configured (no GitHub Actions, Jenkins, etc.).
- **No code linting/formatting tools** configured (no black, pylint, flake8, pre-commit).
- **No type checking** configured (no mypy).

---

## 11. Common Gotchas

1. **Working directory matters**: Scripts expect to be run from the `uq/` directory
   so that relative paths to `config/model_config.yaml` resolve correctly.
2. **OpenMC requires conda**: It's not on PyPI; install via `conda-forge`.
3. **`packing_fraction`** is currently a plain scalar in the config; uncomment the
   `mean`/`relative_stdev`/`pdf` block to make it uncertain.
4. **Two `load_config` functions exist**: one in `uq/util/utils.py` (for general
   use) and one in `uq/openmc_model_run.py` (model-specific).
5. **Path manipulation in tests**: Test files manually add `uq/` to `sys.path` to
   enable imports from the `uq` package.
