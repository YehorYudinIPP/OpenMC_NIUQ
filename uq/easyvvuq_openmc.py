"""
easyvvuq_openmc.py – EasyVVUQ forward uncertainty propagation for OpenMC TBM.

Propagates uncertainties through an OpenMC fixed-source neutron irradiation
model of a Test Blanket Module (TBM) with Li2TiO3 ceramic pebbles.

Uncertain parameters
--------------------
* ``li_ceramic_density``  – Li2TiO3 pebble density (g/cm³)
* ``li6_enrichment``      – Li-6 enrichment in the ceramic (at%)
* ``pebble_radius``       – pebble radius (cm)
* ``packing_fraction``    – pebble packing fraction
* ``li_target_density``   – natural Li target density (g/cm³)
* ``water_density``       – water coolant density (g/cm³, neutron moderation)
* ``graphite_density``    – graphite shielding density (g/cm³, neutron moderation)

Quantities of interest (QoIs)
------------------------------
* ``tritium_production_rate`` – tritium nuclei produced per source neutron (TBR)
* ``total_neutron_flux``      – total neutron flux intensity (per source neutron)

Supports two UQ methods:
  * Polynomial Chaos Expansion (PCE) via ``uq_scheme: pce``
  * Quasi-Monte Carlo (QMC) via ``uq_scheme: qmc``

Usage
-----
    # Run with the default config located in uq/config/model_config.yaml
    python easyvvuq_openmc.py

    # Run with a custom config
    python easyvvuq_openmc.py --config path/to/config.yaml

    # Select UQ scheme and polynomial order explicitly
    python easyvvuq_openmc.py --uq-scheme pce --p-order 2
    python easyvvuq_openmc.py --uq-scheme qmc --n-samples 256
"""

import argparse
import os
import pickle
import sys
from datetime import datetime

import chaospy as cp
import easyvvuq as uq
import numpy as np
from easyvvuq.actions import (
    Actions,
    CreateRunDirectory,
    Decode,
    Encode,
    ExecuteLocal,
)

# ── Path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

parent_dir = os.path.dirname(_here)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ── Local imports ─────────────────────────────────────────────────────────────
from util.Encoder import AdvancedYAMLEncoder
from util.utils import (
    add_timestamp_to_filename,
    load_config,
    validate_execution_setup,
)


# ---------------------------------------------------------------------------
# Parameter & distribution definitions
# ---------------------------------------------------------------------------

def define_model_parameters():
    """
    Return the EasyVVUQ parameter dictionary and the list of QoI column names.

    The parameters dict maps each uncertain TBM input to its EasyVVUQ type
    specification and default value.
    """
    parameters = {
        "li_ceramic_density": {"type": "float", "default": 3.43},
        "li6_enrichment":     {"type": "float", "default": 7.5},
        "pebble_radius":      {"type": "float", "default": 0.1},
        "packing_fraction":   {"type": "float", "default": 0.3},
        "li_target_density":  {"type": "float", "default": 0.534},
        "water_density":      {"type": "float", "default": 1.0},
        "graphite_density":   {"type": "float", "default": 2.1},
    }

    # QoI column names must match the CSV header written by openmc_model_run.py
    qois = ["tritium_production_rate", "total_neutron_flux"]

    return parameters, qois


def define_parameter_distributions(config, cov_override=None, dist_override=None):
    """
    Build a chaospy distribution for each uncertain parameter.

    Parameters
    ----------
    config : dict
        Loaded YAML configuration.
    cov_override : float or None
        If given, apply this coefficient of variation (CoV) to all params.
    dist_override : str or None
        If given, use this distribution family for all params.

    Returns
    -------
    dict
        Mapping from parameter name to chaospy distribution.
    """
    geom = config.get('geometry', {})
    mat = config.get('materials', {})

    def _v(node, key):
        """Extract sub-dict under *key* from *node* (which may be a dict)."""
        return node.get(key, {}) if isinstance(node, dict) else {}

    spec = {
        "li_ceramic_density": _v(_v(mat, 'li_ceramic'), 'density'),
        "li6_enrichment":     _v(_v(mat, 'li_ceramic'), 'li6_enrichment'),
        "pebble_radius":      _v(geom, 'pebble_radius'),
        "packing_fraction":   _v(geom, 'packing_fraction'),
        "li_target_density":  _v(_v(mat, 'li_target'), 'density'),
        "water_density":      _v(_v(mat, 'water'), 'density'),
        "graphite_density":   _v(_v(mat, 'graphite'), 'density'),
    }

    _dist_map = {
        "normal":    cp.Normal,
        "uniform":   cp.Uniform,
        "lognormal": cp.LogNormal,
    }
    _expansion = {"normal": 1.0, "uniform": np.sqrt(3)}

    distributions = {}
    for name, s in spec.items():
        mean = float(s.get('mean', 1.0))
        rel_std = float(cov_override if cov_override is not None
                        else s.get('relative_stdev', 0.05))
        dist_name = (dist_override if dist_override is not None
                     else s.get('pdf', 'normal'))

        if dist_name not in _dist_map:
            raise ValueError(
                f"Unsupported distribution '{dist_name}' for parameter '{name}'. "
                f"Supported: {list(_dist_map)}"
            )

        k = _expansion.get(dist_name, 1.0)
        if dist_name == 'uniform':
            lo = mean * (1.0 - k * rel_std)
            hi = mean * (1.0 + k * rel_std)
            distributions[name] = cp.Uniform(lo, hi)
        elif dist_name == 'normal':
            distributions[name] = cp.Normal(mean, mean * rel_std)
        elif dist_name == 'lognormal':
            distributions[name] = cp.LogNormal(np.log(mean), rel_std)

    print(f" >>> Parameter distributions: {distributions}")
    return distributions


# ---------------------------------------------------------------------------
# Campaign preparation
# ---------------------------------------------------------------------------

def prepare_uq_campaign(config, config_file, fixed_params=None, uq_params=None):
    """
    Build and return a fully configured EasyVVUQ campaign.

    Returns
    -------
    tuple : (campaign, qois, distributions, timestamp, sampler)
    """
    parameters, qois = define_model_parameters()

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder = AdvancedYAMLEncoder(
        template_fname=config_file,
        target_filename="config.yaml",
        parameter_map={
            "li_ceramic_density": "materials.li_ceramic.density.mean",
            "li6_enrichment":     "materials.li_ceramic.li6_enrichment.mean",
            "pebble_radius":      "geometry.pebble_radius.mean",
            "packing_fraction":   "geometry.packing_fraction.mean",
            "li_target_density":  "materials.li_target.density.mean",
            "water_density":      "materials.water.density.mean",
            "graphite_density":   "materials.graphite.density.mean",
        },
        type_conversions={
            "li_ceramic_density": float,
            "li6_enrichment":     float,
            "pebble_radius":      float,
            "packing_fraction":   float,
            "li_target_density":  float,
            "water_density":      float,
            "graphite_density":   float,
        },
        fixed_parameters=fixed_params or {},
    )
    print(f"Encoder prepared: {encoder}")

    # ── Decoder ───────────────────────────────────────────────────────────────
    results_file = config.get('output', {}).get('results_file', 'results.csv')
    decoder = uq.decoders.SimpleCSV(
        target_filename=results_file,
        output_columns=qois,
    )
    print(f"Decoder prepared: {decoder}")

    # ── Execution command ─────────────────────────────────────────────────────
    python_exe, script_path = validate_execution_setup()
    execute = ExecuteLocal(f"{python_exe} {script_path} --config config.yaml")

    # ── Actions ───────────────────────────────────────────────────────────────
    actions = Actions(
        CreateRunDirectory('run_dir'),
        Encode(encoder),
        execute,
        Decode(decoder),
    )

    # ── Campaign ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign = uq.Campaign(
        name=f"openmc_campaign_{timestamp}_",
        params=parameters,
        actions=actions,
    )

    # ── Distributions & sampler ───────────────────────────────────────────────
    distributions = define_parameter_distributions(config)

    uq_params = uq_params or {}
    scheme = uq_params.get('uq_scheme', 'pce')

    if scheme == 'pce':
        p_order = uq_params.get('p_order', 1)
        print(f"Using PCE sampler with polynomial order {p_order}")
        sampler = uq.sampling.PCESampler(vary=distributions, polynomial_order=p_order)

    elif scheme == 'qmc':
        n_samples = uq_params.get('n_samples', 128)
        print(f"Using QMC sampler with {n_samples} samples")
        sampler = uq.sampling.QMCSampler(vary=distributions, n_mc_samples=n_samples)

    else:
        raise ValueError(
            f"Unsupported UQ scheme '{scheme}'. Choose 'pce' or 'qmc'."
        )

    campaign.set_sampler(sampler)
    print(f"Campaign prepared. Sampler: {sampler}")

    return campaign, qois, distributions, timestamp, sampler


# ---------------------------------------------------------------------------
# Campaign execution
# ---------------------------------------------------------------------------

def run_uq_campaign(campaign):
    """Execute all runs in the campaign locally and collate results."""
    print(" >> Running UQ campaign (local execution)…")
    campaign.execute().collate()
    print(" >> Execution and collation complete.")
    return campaign


# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------

def analyse_uq_results(campaign, qois, sampler, uq_params=None):
    """
    Apply the appropriate EasyVVUQ analysis and return the results object.
    """
    uq_params = uq_params or {}
    scheme = uq_params.get('uq_scheme', 'pce')

    if scheme == 'pce':
        analysis = uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=qois)
    elif scheme == 'qmc':
        analysis = uq.analysis.QMCAnalysis(sampler=sampler, qoi_cols=qois)
    else:
        raise ValueError(f"Unsupported UQ scheme '{scheme}'.")

    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    print(f"\n >>> Analysis results:\n{results}")

    for qoi in qois:
        print(f"\nStatistics for '{qoi}':")
        print(results.describe(qoi))

    # Persist to disk
    results_file = add_timestamp_to_filename("analysis_results_openmc_uq.pickle")
    with open(results_file, 'wb') as fh:
        pickle.dump(results, fh)
    print(f"Analysis results saved to: {results_file}")

    return results


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def perform_uq_openmc(config_file=None, fixed_params=None, uq_params=None):
    """
    Orchestrate the full EasyVVUQ uncertainty-propagation workflow for OpenMC.

    Parameters
    ----------
    config_file : str or None
        Path to the YAML configuration file.  When *None* the value is taken
        from the ``--config`` command-line argument (default: ``config.yaml``).
    fixed_params : dict or None
        Parameters fixed for every run (not sampled by EasyVVUQ).
    uq_params : dict or None
        UQ method settings, e.g.
        ``{'uq_scheme': 'pce', 'p_order': 2}`` or
        ``{'uq_scheme': 'qmc', 'n_samples': 256}``.
    """
    print("\n ! Starting OpenMC UQ campaign !\n")
    print(f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Resolve configuration file ────────────────────────────────────────────
    if config_file is None:
        parser = argparse.ArgumentParser(
            description='Run EasyVVUQ forward UQ campaign for OpenMC'
        )
        parser.add_argument(
            '--config', '-c',
            default=os.path.join(_here, 'config', 'model_config.yaml'),
            help='Path to YAML configuration file',
        )
        parser.add_argument(
            '--uq-scheme',
            default='pce',
            choices=['pce', 'qmc'],
            help="UQ sampling scheme: 'pce' (default) or 'qmc'",
        )
        parser.add_argument(
            '--p-order',
            type=int,
            default=1,
            help='Polynomial order for PCE (default: 1)',
        )
        parser.add_argument(
            '--n-samples',
            type=int,
            default=128,
            help='Number of samples for QMC (default: 128)',
        )
        args = parser.parse_args()
        config_file = args.config

        if uq_params is None:
            uq_params = {
                'uq_scheme': args.uq_scheme,
                'p_order': args.p_order,
                'n_samples': args.n_samples,
            }

    config = load_config(config_file)
    if config is None:
        print("No valid configuration found – aborting.")
        return

    if uq_params is None:
        uq_params = {'uq_scheme': 'pce', 'p_order': 1}

    print(f" >> UQ parameters: {uq_params}")
    print(f" >> Fixed parameters: {fixed_params}")

    # ── Prepare ───────────────────────────────────────────────────────────────
    campaign, qois, distributions, timestamp, sampler = prepare_uq_campaign(
        config, config_file,
        fixed_params=fixed_params,
        uq_params=uq_params,
    )

    # ── Execute ───────────────────────────────────────────────────────────────
    campaign = run_uq_campaign(campaign)
    campaign.campaign_db.dump()

    # ── Analyse ───────────────────────────────────────────────────────────────
    results = analyse_uq_results(campaign, qois, sampler, uq_params=uq_params)

    # Persist campaign config
    cfg_file = add_timestamp_to_filename("uq_campaign_config.pickle")
    with open(cfg_file, 'wb') as fh:
        pickle.dump(config, fh)
    print(f"Campaign configuration saved to: {cfg_file}")

    print("\nOpenMC UQ campaign completed successfully!")
    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    perform_uq_openmc()
