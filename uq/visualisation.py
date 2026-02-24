"""
visualisation.py – Post-campaign visualisation for OpenMC UQ results.

Generates plots of:
* QoI descriptive statistics (mean ± std, percentile ranges)
* First-order and total Sobol sensitivity indices per input parameter

Inspired by the plotting utilities in FESTIM-NIUQ and the EasyVVUQ
built-in ``plot_sobols_first`` / ``plot_moments`` helpers.

Usage
-----
Called automatically at the end of ``perform_uq_openmc()`` or standalone::

    from uq.visualisation import visualise_results
    visualise_results(results, qois, distributions, output_dir="plots")
"""

import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless environments
import matplotlib.pyplot as plt
import numpy as np


# ── Human-readable labels ────────────────────────────────────────────────────
_QOI_LABELS = {
    "tritium_production_rate": "Tritium Production Rate (TBR)",
    "total_neutron_flux": "Total Neutron Flux",
    "tbm_incident_flux": "TBM Incident Neutron Flux",
    "tbm_inner_flux": "TBM Inner Neutron Flux",
    "tbm_heating": "TBM Nuclear Heating",
    "tbm_neutron_leakage": "TBM Neutron Leakage",
}

_PARAM_LABELS = {
    "li_ceramic_density": r"Li$_2$TiO$_3$ Density",
    "li6_enrichment": r"$^{6}$Li Enrichment",
    "pebble_radius": "Pebble Radius",
    "packing_fraction": "Packing Fraction",
    "graphite_thickness": "Graphite Thickness",
}

_PRIMARY_COLOR = "#4C72B0"
_SECONDARY_COLOR = "#DD8452"


def _label_for_qoi(qoi):
    """Return a human-readable label for a QoI name."""
    return _QOI_LABELS.get(qoi, qoi)


def _label_for_param(param):
    """Return a human-readable label for a parameter name."""
    return _PARAM_LABELS.get(param, param)


# ---------------------------------------------------------------------------
# QoI statistics
# ---------------------------------------------------------------------------

def plot_qoi_statistics(results, qois, output_dir):
    """
    Bar chart showing mean ± std for each QoI, with 10–90 % range whiskers.

    Parameters
    ----------
    results : easyvvuq analysis results
        Object returned by ``campaign.get_last_analysis()``.
    qois : list[str]
        Names of the quantities of interest.
    output_dir : str
        Directory where the figure is saved.

    Returns
    -------
    str
        Path to the saved figure.
    """
    means, stds = [], []
    p10s, p90s = [], []
    labels = []

    print("\n >> Plotting QoI statistics for:")
    print("    " + "\n    ".join(qois))

    for qoi in qois:
        mean_val = float(np.squeeze(results.describe(qoi, "mean")))
        std_val = float(np.squeeze(results.describe(qoi, "std")))
        p10_val = float(np.squeeze(results.describe(qoi, "10%")))
        p90_val = float(np.squeeze(results.describe(qoi, "90%")))

        means.append(mean_val)
        stds.append(std_val)
        p10s.append(p10_val)
        p90s.append(p90_val)
        labels.append(_label_for_qoi(qoi))

    n_qois = len(qois)
    fig, axes = plt.subplots(1, n_qois, figsize=(max(6, 5 * n_qois), 5))
    if n_qois == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.bar([0], [means[i]], yerr=[stds[i]], capsize=8,
               color=_PRIMARY_COLOR, edgecolor="black", alpha=0.85,
               label=r"Mean $\pm$ 1 $\sigma$")

        lower_err = means[i] - p10s[i]
        upper_err = p90s[i] - means[i]
        ax.errorbar([0], [means[i]], yerr=[[lower_err], [upper_err]],
                    fmt="none", ecolor="grey", elinewidth=1.5, capsize=5,
                    label="10 %–90 % range")

        ax.set_xticks([0])
        ax.set_xticklabels([labels[i]], fontsize=11)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(labels[i], fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()

    filepath = os.path.join(output_dir, "qoi_statistics.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ QoI statistics plot saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Sobol indices
# ---------------------------------------------------------------------------

def plot_sobol_indices(results, qois, distributions, output_dir):
    """
    Grouped bar chart of first-order and total Sobol indices for each QoI.

    Parameters
    ----------
    results : easyvvuq analysis results
        Object returned by ``campaign.get_last_analysis()``.
    qois : list[str]
        QoI column names.
    distributions : dict
        Mapping of parameter name → chaospy distribution (used for ordering).
    output_dir : str
        Directory where figures are saved.

    Returns
    -------
    list[str]
        Paths to the saved figures (one per QoI).
    """
    param_names = list(distributions.keys())
    param_labels = [_label_for_param(p) for p in param_names]

    saved = []
    for qoi in qois:
        sobols_first = results.sobols_first(qoi)

        # Attempt to get total Sobol indices (PCE provides them; QMC may not)
        try:
            sobols_total = results.sobols_total(qoi)
        except (AttributeError, RuntimeError):
            sobols_total = None

        s1_vals = []
        st_vals = []
        for p in param_names:
            s1 = sobols_first.get(p, [0.0])
            s1_vals.append(float(np.squeeze(s1)))
            if sobols_total is not None:
                st = sobols_total.get(p, [0.0])
                st_vals.append(float(np.squeeze(st)))

        x = np.arange(len(param_names))
        has_total = len(st_vals) == len(param_names)
        width = 0.35 if has_total else 0.5

        fig, ax = plt.subplots(figsize=(max(8, 2 * len(param_names)), 5))

        if has_total:
            ax.bar(x - width / 2, s1_vals, width, label="First-order",
                   color=_PRIMARY_COLOR, edgecolor="black", alpha=0.85)
            ax.bar(x + width / 2, st_vals, width, label="Total",
                   color=_SECONDARY_COLOR, edgecolor="black", alpha=0.85)
        else:
            ax.bar(x, s1_vals, width, label="First-order",
                   color=_PRIMARY_COLOR, edgecolor="black", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, fontsize=10, rotation=25, ha="right")
        ax.set_ylabel("Sobol Index", fontsize=12)
        ax.set_title(f"Sobol Indices – {_label_for_qoi(qoi)}", fontsize=14)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()

        filename = f"sobol_indices_{qoi}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved.append(filepath)
        print(f"  ✓ Sobol indices plot saved: {filepath}")

    return saved


# ---------------------------------------------------------------------------
# QoI distribution (histogram from raw samples)
# ---------------------------------------------------------------------------

def plot_qoi_distributions(results, qois, output_dir):
    """
    Histogram of QoI sample values (if raw data is available).

    Parameters
    ----------
    results : easyvvuq analysis results
        Object returned by ``campaign.get_last_analysis()``.
    qois : list[str]
        QoI column names.
    output_dir : str
        Directory where the figure is saved.

    Returns
    -------
    str or None
        Path to the saved figure, or *None* if raw data is unavailable.
    """
    raw = getattr(results, "raw_data", None)
    if raw is None:
        print("  ⓘ No raw sample data available – skipping distribution plot.")
        return None

    n_qois = len(qois)
    fig, axes = plt.subplots(1, n_qois, figsize=(6 * n_qois, 5))
    if n_qois == 1:
        axes = [axes]

    for ax, qoi in zip(axes, qois):
        try:
            samples = np.squeeze(np.array(raw[qoi]))
            ax.hist(samples, bins="auto", color=_PRIMARY_COLOR, edgecolor="black",
                    alpha=0.75)
            ax.set_xlabel(_label_for_qoi(qoi), fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"Distribution of {_label_for_qoi(qoi)}", fontsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        except (KeyError, TypeError):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="grey")
            ax.set_title(_label_for_qoi(qoi))

    fig.tight_layout()
    filepath = os.path.join(output_dir, "qoi_distributions.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ QoI distribution plot saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def visualise_results(results, qois, distributions, output_dir=None,
                      timestamp=None):
    """
    Generate all standard post-campaign visualisation plots.

    Parameters
    ----------
    results : easyvvuq analysis results
        Analysis results object from ``campaign.get_last_analysis()``.
    qois : list[str]
        QoI column names.
    distributions : dict
        Parameter name → chaospy distribution mapping.
    output_dir : str or None
        Target directory for plots.  Created automatically when it does not
        exist.  Defaults to ``plots_openmc_uq_<timestamp>/``.
    timestamp : str or None
        Timestamp string appended to the default directory name.

    Returns
    -------
    list[str]
        Paths to all generated plot files.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = f"plots_openmc_uq_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n >> Generating visualisation plots in: {output_dir}")

    saved_files = []

    # 1. QoI statistics (mean ± std, percentile range)
    saved_files.append(plot_qoi_statistics(results, qois, output_dir))

    # 2. Sobol indices per input parameter
    saved_files.extend(plot_sobol_indices(results, qois, distributions,
                                         output_dir))

    # 3. QoI distributions (histograms from raw samples, if available)
    dist_path = plot_qoi_distributions(results, qois, output_dir)
    if dist_path is not None:
        saved_files.append(dist_path)

    print(f" >> Visualisation complete. {len(saved_files)} plot(s) generated.")
    return saved_files
