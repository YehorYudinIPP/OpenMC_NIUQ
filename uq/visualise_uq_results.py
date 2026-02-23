"""
visualise_uq_results.py – Standalone post-campaign visualisation script.

Loads previously saved analysis results (pickle) and regenerates all
visualisation plots without re-running the UQ campaign.

Usage
-----
    # Provide the analysis results pickle (required)
    python visualise_uq_results.py --results analysis_results_openmc_uq_20250720_143025.pickle

    # Optionally provide the campaign config pickle for distribution info
    python visualise_uq_results.py --results analysis_results_openmc_uq_20250720_143025.pickle \
                                   --config uq_campaign_config_20250720_143025.pickle

    # Specify a custom output directory
    python visualise_uq_results.py --results analysis_results_openmc_uq_20250720_143025.pickle \
                                   --output-dir my_plots
"""

import argparse
import os
import pickle
import sys
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from visualisation import visualise_results

# ── Default QoI and parameter names (same as easyvvuq_openmc.py) ─────────────
_DEFAULT_QOIS = ["tritium_production_rate", "total_neutron_flux"]
_DEFAULT_PARAM_NAMES = [
    "li_ceramic_density",
    "li6_enrichment",
    "pebble_radius",
    "packing_fraction",
    "graphite_thickness",
]


def load_pickle(filepath):
    """Load and return a Python object from a pickle file."""
    with open(filepath, "rb") as fh:
        return pickle.load(fh)


def _distributions_from_config(config):
    """
    Reconstruct the parameter-name → distribution mapping from a saved
    campaign config dict.  Only the *keys* (and their order) are used by
    the plotting routines, so the values are set to ``None`` when the
    optional ``chaospy`` dependency is not available.
    """
    try:
        from easyvvuq_openmc import define_parameter_distributions
        return define_parameter_distributions(config)
    except Exception:
        # Fall back: extract parameter names from the config structure
        param_names = list(_extract_param_names(config))
        return {name: None for name in param_names}


def _extract_param_names(config):
    """
    Yield uncertain parameter names found in a campaign config dict.

    Uncertain parameters are identified by having a nested ``mean`` key.
    """
    geom = config.get("geometry", {})
    mat = config.get("materials", {})
    li_ceramic = mat.get("li_ceramic", {}) if isinstance(mat, dict) else {}

    _mapping = {
        "li_ceramic_density": li_ceramic.get("density", {}),
        "li6_enrichment": li_ceramic.get("li6_enrichment", {}),
        "pebble_radius": geom.get("pebble_radius", {}),
        "packing_fraction": geom.get("packing_fraction", {}),
        "graphite_thickness": geom.get("graphite_thickness", {}),
    }
    for name, node in _mapping.items():
        if isinstance(node, dict) and "mean" in node:
            yield name


def main(results_file, config_file=None, output_dir=None):
    """
    Generate visualisation plots from saved UQ campaign results.

    Parameters
    ----------
    results_file : str
        Path to the pickled EasyVVUQ analysis results
        (``analysis_results_openmc_uq_*.pickle``).
    config_file : str or None
        Path to the pickled campaign configuration
        (``uq_campaign_config_*.pickle``).  When provided, parameter
        distributions are reconstructed from it; otherwise the default
        parameter ordering is used.
    output_dir : str or None
        Target directory for the plot files.  When *None* a new
        timestamped folder ``plots_openmc_uq_<timestamp>/`` is created.
    """
    print(f"\n >> Loading analysis results from: {results_file}")
    results = load_pickle(results_file)

    qois = _DEFAULT_QOIS

    # Build a distributions-like dict.  Only the keys (parameter names and
    # their order) are used by the plotting routines.
    if config_file is not None:
        print(f" >> Loading campaign config from: {config_file}")
        config = load_pickle(config_file)
        distributions = _distributions_from_config(config)
    else:
        distributions = {name: None for name in _DEFAULT_PARAM_NAMES}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = visualise_results(
        results, qois, distributions,
        output_dir=output_dir, timestamp=timestamp,
    )

    print(f"\n >> Done. {len(saved)} plot(s) saved.")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone visualisation of OpenMC UQ campaign results",
    )
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="Path to the analysis results pickle file "
             "(analysis_results_openmc_uq_*.pickle)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to the campaign config pickle file "
             "(uq_campaign_config_*.pickle).  Optional.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for plot images.  "
             "Defaults to plots_openmc_uq_<timestamp>/.",
    )
    args = parser.parse_args()
    main(args.results, config_file=args.config, output_dir=args.output_dir)
