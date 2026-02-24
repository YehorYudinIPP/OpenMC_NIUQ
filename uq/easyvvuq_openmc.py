"""
easyvvuq_openmc.py – EasyVVUQ forward uncertainty propagation for OpenMC TBM.

Propagates uncertainties through an OpenMC fixed-source neutron irradiation
model of a Test Blanket Module (TBM) with Li2TiO3 ceramic pebbles.

Uncertain parameters are **auto-discovered** from the YAML configuration file.
Any parameter specified as a dictionary with ``mean``, ``relative_stdev``, and
``pdf`` keys is treated as uncertain.  Plain scalar values are fixed.

If a parameter is a plain scalar but is listed in the ``vary`` set (or if a
distribution dict is incomplete), a warning is emitted and a Uniform
distribution with CoV = 0.05 is applied by default.

Quantities of interest (QoIs) are read from ``output.qoi`` in the config.

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

Adding new uncertain parameters
-------------------------------
To add a new uncertain parameter to the UQ campaign you only need to modify
the YAML configuration file (``model_config.yaml``):

1. Express the parameter as a dictionary with ``mean``, ``relative_stdev``,
   and ``pdf`` keys instead of a plain scalar value.
2. In ``openmc_model_run.py``, read the parameter value through the
   ``_get_mean()`` helper so that both scalar and distribution-dict formats
   are handled transparently.

The functions ``discover_uncertain_parameters()``,
``define_parameter_distributions()``, and ``prepare_uq_campaign()`` in this
module will automatically pick up the new parameter from the config — **no
code changes are needed in this file**.
"""

import argparse
import logging
import os
import pickle
import sys
import warnings
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
from visualisation import visualise_results

# ── Logging setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger("openmc_uq")

_DEFAULT_COV = 0.05
_DEFAULT_PDF = "uniform"


def setup_logging(log_file=None):
    """
    Configure logging to write to both the console and a file.

    Parameters
    ----------
    log_file : str or None
        Path to the log file.  When *None* a timestamped name is generated.

    Returns
    -------
    str
        Path to the log file.
    """
    if log_file is None:
        log_file = add_timestamp_to_filename("openmc_uq_campaign.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove pre-existing handlers to avoid duplicate output on repeated calls
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (DEBUG and above – captures everything)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logger.info("Logging initialised.  Log file: %s", log_file)
    return log_file


# ---------------------------------------------------------------------------
# Auto-discovery of uncertain parameters from config
# ---------------------------------------------------------------------------

def _walk_config(node, prefix=""):
    """
    Recursively yield ``(dot_path, value)`` for every leaf in *node*.

    A leaf is either a scalar or a dict that contains the special key
    ``mean`` (i.e. a UQ-spec node).
    """
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "mean" in value:
            # This is an uncertain-parameter spec node
            yield path, value
        elif isinstance(value, dict):
            yield from _walk_config(value, prefix=path)


def discover_uncertain_parameters(config):
    """
    Scan the config dict and return information about every uncertain
    parameter (those specified with ``mean`` / ``relative_stdev`` / ``pdf``).

    Returns
    -------
    list[dict]
        Each element has keys ``name``, ``path``, ``mean``, ``relative_stdev``,
        ``pdf``.  ``name`` is the leaf key (e.g. ``'density'`` becomes
        ``'li_ceramic_density'`` only after the caller applies a naming
        convention).
    """
    # Sections we scan for uncertain parameters
    found = []
    for dot_path, spec in _walk_config(config):
        mean = float(spec.get("mean", 0.0))
        rel_std = float(spec.get("relative_stdev", _DEFAULT_COV))
        pdf = spec.get("pdf", None)

        if pdf is None:
            warnings.warn(
                f"Parameter at '{dot_path}' has a 'mean' key but no 'pdf'. "
                f"Applying default Uniform distribution with CoV={_DEFAULT_COV}.",
                stacklevel=2,
            )
            pdf = _DEFAULT_PDF

        # Derive a short parameter name from the YAML path.
        # Convention: join the last two path components with '_'
        # e.g. "materials.li_ceramic.density" -> "li_ceramic_density"
        parts = dot_path.split(".")
        if len(parts) >= 2:
            short_name = f"{parts[-2]}_{parts[-1]}"
        else:
            short_name = parts[-1]

        found.append({
            "name": short_name,
            "path": dot_path,
            "mean": mean,
            "relative_stdev": rel_std,
            "pdf": pdf,
        })

    logger.info("Auto-discovered %d uncertain parameter(s) from config.", len(found))
    for p in found:
        logger.info("  • %s  (path=%s, mean=%.4g, CoV=%.3g, pdf=%s)",
                     p["name"], p["path"], p["mean"], p["relative_stdev"], p["pdf"])
    return found


# ---------------------------------------------------------------------------
# Parameter & distribution definitions
# ---------------------------------------------------------------------------

def define_model_parameters(config):
    """
    Return the EasyVVUQ parameter dictionary and the list of QoI column names,
    both derived entirely from the YAML configuration file.

    Parameters that are specified with ``mean`` / ``relative_stdev`` / ``pdf``
    in the config are treated as uncertain; all others are fixed.
    """
    uncertain = discover_uncertain_parameters(config)

    parameters = {}
    for p in uncertain:
        parameters[p["name"]] = {"type": "float", "default": p["mean"]}

    # QoI column names – read from config (fall back to original two)
    qois = config.get("output", {}).get("qoi", [
        "tritium_production_rate",
        "total_neutron_flux",
    ])

    return parameters, qois, uncertain


def define_parameter_distributions(config, uncertain_specs=None,
                                   cov_override=None, dist_override=None):
    """
    Build a chaospy distribution for each uncertain parameter.

    Parameters
    ----------
    config : dict
        Loaded YAML configuration.
    uncertain_specs : list[dict] or None
        Output of ``discover_uncertain_parameters()``.  When *None* the
        discovery is run internally.
    cov_override : float or None
        If given, apply this coefficient of variation (CoV) to all params.
    dist_override : str or None
        If given, use this distribution family for all params.

    Returns
    -------
    dict
        Mapping from parameter name to chaospy distribution.
    """
    if uncertain_specs is None:
        uncertain_specs = discover_uncertain_parameters(config)

    _dist_map = {
        "normal":    cp.Normal,
        "uniform":   cp.Uniform,
        "lognormal": cp.LogNormal,
    }
    _expansion = {"normal": 1.0, "uniform": np.sqrt(3)}

    distributions = {}
    for spec in uncertain_specs:
        name = spec["name"]
        mean = spec["mean"]
        rel_std = float(cov_override if cov_override is not None
                        else spec["relative_stdev"])
        dist_name = (dist_override if dist_override is not None
                     else spec["pdf"])

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

    logger.info("Parameter distributions: %s", distributions)
    return distributions


# ---------------------------------------------------------------------------
# Campaign preparation
# ---------------------------------------------------------------------------

def prepare_uq_campaign(config, config_file, fixed_params=None, uq_params=None):
    """
    Build and return a fully configured EasyVVUQ campaign.

    The uncertain parameters, their encoder paths, and QoIs are all
    derived from the YAML config – no hard-coded parameter lists.

    Returns
    -------
    tuple : (campaign, qois, distributions, timestamp, sampler)
    """
    parameters, qois, uncertain_specs = define_model_parameters(config)

    # ── Build encoder mappings automatically ──────────────────────────────────
    # Each uncertain spec carries its YAML dot-path ending in the leaf key.
    # The encoder needs the path to the ``mean`` sub-key inside that dict.
    parameter_map = {}
    type_conversions = {}
    for spec in uncertain_specs:
        parameter_map[spec["name"]] = f"{spec['path']}.mean"
        type_conversions[spec["name"]] = float

    # ── Encoder ───────────────────────────────────────────────────────────────
    encoder = AdvancedYAMLEncoder(
        template_fname=config_file,
        target_filename="config.yaml",
        parameter_map=parameter_map,
        type_conversions=type_conversions,
        fixed_parameters=fixed_params or {},
    )
    logger.info("Encoder prepared with parameter map: %s", parameter_map)

    # ── Decoder ───────────────────────────────────────────────────────────────
    results_file = config.get('output', {}).get('results_file', 'results.csv')
    decoder = uq.decoders.SimpleCSV(
        target_filename=results_file,
        output_columns=qois,
    )
    logger.info("Decoder prepared for QoIs: %s", qois)

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
    distributions = define_parameter_distributions(
        config, uncertain_specs=uncertain_specs)

    uq_params = uq_params or {}
    scheme = uq_params.get('uq_scheme', 'pce')

    if scheme == 'pce':
        p_order = uq_params.get('p_order', 1)
        logger.info("Using PCE sampler with polynomial order %d", p_order)
        sampler = uq.sampling.PCESampler(vary=distributions, polynomial_order=p_order)

    elif scheme == 'qmc':
        n_samples = uq_params.get('n_samples', 128)
        logger.info("Using QMC sampler with %d samples", n_samples)
        sampler = uq.sampling.QMCSampler(vary=distributions, n_mc_samples=n_samples)

    else:
        raise ValueError(
            f"Unsupported UQ scheme '{scheme}'. Choose 'pce' or 'qmc'."
        )

    campaign.set_sampler(sampler)
    logger.info("Campaign prepared. Sampler: %s", sampler)

    return campaign, qois, distributions, timestamp, sampler


# ---------------------------------------------------------------------------
# Campaign execution
# ---------------------------------------------------------------------------

def run_uq_campaign(campaign):
    """Execute all runs in the campaign locally and collate results."""
    logger.info("Running UQ campaign (local execution)…")
    campaign.execute().collate()
    logger.info("Execution and collation complete.")
    return campaign


# ---------------------------------------------------------------------------
# Results analysis
# ---------------------------------------------------------------------------

def analyse_uq_results(campaign, qois, sampler, distributions,
                       uq_params=None):
    """
    Apply the appropriate EasyVVUQ analysis, print final UQ & SA results
    to the console / log, and return the results object.
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

    # ── Print final UQ & SA results ──────────────────────────────────────────
    param_names = list(distributions.keys())

    logger.info("")
    logger.info("=" * 70)
    logger.info("  FINAL UQ & SA RESULTS")
    logger.info("=" * 70)

    for qoi in qois:
        logger.info("")
        logger.info("─── QoI: %s ───", qoi)
        desc = results.describe(qoi)
        logger.info("  Descriptive statistics:\n%s", desc)

        # First-order Sobol indices
        try:
            s1 = results.sobols_first(qoi)
            logger.info("  First-order Sobol indices:")
            for p in param_names:
                val = float(np.squeeze(s1.get(p, [0.0])))
                logger.info("    %-25s  S1 = %.4f", p, val)
        except Exception:
            logger.info("  First-order Sobol indices: not available")

        # Total Sobol indices (PCE only)
        try:
            st = results.sobols_total(qoi)
            logger.info("  Total Sobol indices:")
            for p in param_names:
                val = float(np.squeeze(st.get(p, [0.0])))
                logger.info("    %-25s  ST = %.4f", p, val)
        except (AttributeError, RuntimeError):
            logger.info("  Total Sobol indices: not available (QMC scheme)")

    logger.info("")
    logger.info("=" * 70)

    # Persist to disk
    results_file = add_timestamp_to_filename("analysis_results_openmc_uq.pickle")
    with open(results_file, 'wb') as fh:
        pickle.dump(results, fh)
    logger.info("Analysis results saved to: %s", results_file)

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
    # ── Set up logging early ─────────────────────────────────────────────────
    log_file = setup_logging()

    logger.info("")
    logger.info("! Starting OpenMC UQ campaign !")
    logger.info("time: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

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
        logger.error("No valid configuration found – aborting.")
        return

    if uq_params is None:
        uq_params = {'uq_scheme': 'pce', 'p_order': 1}

    logger.info("UQ parameters: %s", uq_params)
    logger.info("Fixed parameters: %s", fixed_params)

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
    results = analyse_uq_results(
        campaign, qois, sampler, distributions, uq_params=uq_params)

    # ── Visualise ─────────────────────────────────────────────────────────────
    visualise_results(results, qois, distributions, timestamp=timestamp)

    # Persist campaign config
    cfg_file = add_timestamp_to_filename("uq_campaign_config.pickle")
    with open(cfg_file, 'wb') as fh:
        pickle.dump(config, fh)
    logger.info("Campaign configuration saved to: %s", cfg_file)

    logger.info("")
    logger.info("OpenMC UQ campaign completed successfully!")
    logger.info("Full log saved to: %s", log_file)
    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    perform_uq_openmc()
