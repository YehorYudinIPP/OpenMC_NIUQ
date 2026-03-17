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
from scipy.stats import gaussian_kde


# ── Human-readable labels ────────────────────────────────────────────────────
_QOI_LABELS = {
    "tritium_production_rate": "Tritium Production Rate (TBR)",
    "total_neutron_flux": "Total Neutron Flux",
    "tbm_incident_flux": "TBM Incident Neutron Flux",
    "tbm_inner_flux": "TBM Inner Neutron Flux",
    "tbm_heating": "TBM Nuclear Heating",
    "tbm_neutron_leakage": "TBM Neutron Leakage",
}

_QOI_UNITS = {
    "tritium_production_rate": "T/source neutron",
    "total_neutron_flux": "n/cm²/s per source",
    "tbm_incident_flux": "n/cm²/s",
    "tbm_inner_flux": "n/cm²/s",
    "tbm_heating": "eV/source neutron",
    "tbm_neutron_leakage": "n/cm²/s",
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


def _unit_for_qoi(qoi):
    """Return the physical unit string for a QoI name."""
    return _QOI_UNITS.get(qoi, "")


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
        unit = _unit_for_qoi(qois[i])
        ax.set_ylabel(f"[{unit}]" if unit else "Value", fontsize=12)
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
            unit = _unit_for_qoi(qoi)
            xlabel = f"{_label_for_qoi(qoi)} [{unit}]" if unit else _label_for_qoi(qoi)
            ax.set_xlabel(xlabel, fontsize=11)
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
# Second-order Sobol indices heatmap
# ---------------------------------------------------------------------------

def plot_sobol_second_order_heatmap(results, qois, distributions, output_dir):
    """
    Heatmap of second-order Sobol indices for each QoI.

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
        Paths to the saved figures (one per QoI), or empty list if
        second-order indices are unavailable.
    """
    param_names = list(distributions.keys())
    param_labels = [_label_for_param(p) for p in param_names]
    n = len(param_names)

    saved = []
    for qoi in qois:
        try:
            sobols_second = results.sobols_second(qoi)
        except (AttributeError, RuntimeError):
            print("  ⓘ Second-order Sobol indices unavailable – "
                  "skipping heatmap.")
            return saved

        # First-order indices for the main diagonal
        try:
            sobols_first = results.sobols_first(qoi)
        except (AttributeError, RuntimeError):
            sobols_first = {}

        matrix = np.zeros((n, n))
        for i, pi in enumerate(param_names):
            # Diagonal: first-order Sobol index
            s1 = sobols_first.get(pi, [0.0])
            matrix[i, i] = float(np.squeeze(s1))
            # Off-diagonal: second-order Sobol indices
            row = sobols_second.get(pi, {})
            for j, pj in enumerate(param_names):
                if i != j:
                    val = row.get(pj, [0.0])
                    matrix[i, j] = float(np.squeeze(val))

        fig, ax = plt.subplots(figsize=(max(6, 1.2 * n), max(5, 1.0 * n)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(param_labels, fontsize=9, rotation=35, ha="right")
        ax.set_yticklabels(param_labels, fontsize=9)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() * 0.6
                        else "black")

        ax.set_title(
            f"Second-Order Sobol Indices – {_label_for_qoi(qoi)}", fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        filename = f"sobol_second_order_{qoi}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved.append(filepath)
        print(f"  ✓ Second-order Sobol heatmap saved: {filepath}")

    return saved


# ---------------------------------------------------------------------------
# First-order Sobol pie chart
# ---------------------------------------------------------------------------

def plot_sobol_first_order_pie(results, qois, distributions, output_dir):
    """
    Pie chart of first-order Sobol indices for each QoI, with a segment
    representing higher-order interactions (1 − Σ S₁).

    Parameters
    ----------
    results : easyvvuq analysis results
        Object returned by ``campaign.get_last_analysis()``.
    qois : list[str]
        QoI column names.
    distributions : dict
        Mapping of parameter name → chaospy distribution.
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
        s1_vals = []
        for p in param_names:
            s1 = sobols_first.get(p, [0.0])
            s1_vals.append(max(0.0, float(np.squeeze(s1))))

        total_first = sum(s1_vals)
        higher_order = max(0.0, 1.0 - total_first)

        sizes = s1_vals + [higher_order]
        labels = param_labels + ["Higher-Order"]
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(
            sizes, autopct="%1.1f%%",
            colors=colors, startangle=140,
            pctdistance=0.85, textprops={"fontsize": 9})
        for at in autotexts:
            at.set_fontsize(8)
        ax.legend(wedges, labels, title="Parameters", loc="center left",
                  bbox_to_anchor=(1.0, 0.5), fontsize=9)
        ax.set_title(
            f"First-Order Sobol Indices – {_label_for_qoi(qoi)}", fontsize=13)
        fig.tight_layout()

        filename = f"sobol_pie_{qoi}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved.append(filepath)
        print(f"  ✓ Sobol pie chart saved: {filepath}")

    return saved


# ---------------------------------------------------------------------------
# QoI statistics table
# ---------------------------------------------------------------------------

def plot_qoi_statistics_table(results, qois, output_dir):
    """
    Render a table with Mean and Standard Deviation for each QoI.

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
    str
        Path to the saved figure.
    """
    rows = []
    for qoi in qois:
        mean_val = float(np.squeeze(results.describe(qoi, "mean")))
        std_val = float(np.squeeze(results.describe(qoi, "std")))
        unit = _unit_for_qoi(qoi)
        rows.append([_label_for_qoi(qoi), unit, f"{mean_val:.4e}", f"{std_val:.4e}"])

    fig, ax = plt.subplots(
        figsize=(max(8, 1.8 * len(qois)), 1.0 + 0.45 * len(qois)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Quantity of Interest", "Units", "Mean", "Std Dev"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(4)))
    table.scale(1, 1.4)

    # Style header row
    for j in range(4):
        table[0, j].set_facecolor(_PRIMARY_COLOR)
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("QoI Summary Statistics", fontsize=14, pad=20)
    fig.tight_layout()

    filepath = os.path.join(output_dir, "qoi_statistics_table.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ QoI statistics table saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Relative standard deviation bar plot (log-scale)
# ---------------------------------------------------------------------------

def plot_relative_std(results, qois, output_dir):
    """
    Bar plot of relative standard deviation (CoV = std / |mean|) for each
    QoI, displayed on a logarithmic scale.

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
    str
        Path to the saved figure.
    """
    labels, rel_stds = [], []
    for qoi in qois:
        mean_val = float(np.squeeze(results.describe(qoi, "mean")))
        std_val = float(np.squeeze(results.describe(qoi, "std")))
        if mean_val != 0:
            rel_stds.append(abs(std_val / mean_val))
        else:
            rel_stds.append(0.0)
        labels.append(_label_for_qoi(qoi))

    x = np.arange(len(qois))
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(qois)), 5))
    ax.bar(x, rel_stds, color=_PRIMARY_COLOR, edgecolor="black", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Relative Std Dev (CoV)", fontsize=12)
    ax.set_title("Relative Standard Deviation of QoIs", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    filepath = os.path.join(output_dir, "qoi_relative_std.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ Relative std dev plot saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Input uncertainty PDFs with KDE fit
# ---------------------------------------------------------------------------

def plot_input_uncertainty_pdfs(distributions, output_dir, n_samples=10000):
    """
    Plot the probability density function of each uncertain input
    parameter by sampling from its distribution and fitting a KDE.

    Parameters
    ----------
    distributions : dict
        Mapping of parameter name → chaospy distribution.
        If a value is ``None`` the parameter is skipped.
    output_dir : str
        Directory where the figure is saved.
    n_samples : int
        Number of samples drawn from each distribution for the KDE.

    Returns
    -------
    str or None
        Path to the saved figure, or *None* if no plottable distributions
        are available.
    """
    plottable = {k: v for k, v in distributions.items() if v is not None}
    if not plottable:
        print("  ⓘ No distribution objects available – skipping input PDF plot.")
        return None

    n = len(plottable)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, dist) in zip(axes, plottable.items()):
        samples = np.squeeze(np.array(dist.sample(n_samples)))
        x_grid = np.linspace(samples.min(), samples.max(), 300)

        # Prefer analytical PDF when available; fall back to KDE
        try:
            y_vals = np.squeeze(np.array(dist.pdf(x_grid)))
            label = "PDF"
        except Exception:
            kde = gaussian_kde(samples)
            y_vals = kde(x_grid)
            label = "KDE fit"

        ax.plot(x_grid, y_vals, color=_PRIMARY_COLOR, linewidth=2,
                label=label)
        ax.fill_between(x_grid, y_vals, alpha=0.25, color=_PRIMARY_COLOR)
        ax.set_xlabel(_label_for_param(name), fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Input PDF – {_label_for_param(name)}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    filepath = os.path.join(output_dir, "input_uncertainty_pdfs.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ Input uncertainty PDF plot saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# QoI PDF from PCE surrogate (KDE of raw samples)
# ---------------------------------------------------------------------------

def plot_qoi_pdf_pce(results, qois, output_dir):
    """
    Plot the estimated probability density function of each QoI derived
    from the PCE surrogate model.

    The PDF is obtained by fitting a Kernel Density Estimate (KDE) to the
    raw QoI samples stored in the analysis results.

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
        print("  ⓘ No raw sample data available – skipping QoI PDF plot.")
        return None

    n_qois = len(qois)
    fig, axes = plt.subplots(1, n_qois, figsize=(6 * n_qois, 5))
    if n_qois == 1:
        axes = [axes]

    for ax, qoi in zip(axes, qois):
        try:
            samples = np.squeeze(np.array(raw[qoi]))
            if samples.ndim == 0 or len(samples) < 2:
                raise ValueError("Not enough samples for KDE")
            kde = gaussian_kde(samples)
            x_grid = np.linspace(samples.min(), samples.max(), 300)

            ax.plot(x_grid, kde(x_grid), color=_PRIMARY_COLOR, linewidth=2,
                    label="PCE PDF")
            ax.fill_between(x_grid, kde(x_grid), alpha=0.25,
                            color=_PRIMARY_COLOR)
            unit = _unit_for_qoi(qoi)
            xlabel = (f"{_label_for_qoi(qoi)} [{unit}]" if unit
                      else _label_for_qoi(qoi))
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"QoI PDF – {_label_for_qoi(qoi)}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        except (KeyError, TypeError, ValueError):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="grey")
            ax.set_title(_label_for_qoi(qoi))

    fig.tight_layout()
    filepath = os.path.join(output_dir, "qoi_pdf_pce.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ QoI PDF (PCE) plot saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# QoI individual run values (strip / scatter plot)
# ---------------------------------------------------------------------------

def plot_qoi_individual_runs(results, qois, output_dir):
    """
    Scatter plot showing the value of each QoI for every individual run.

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
        print("  ⓘ No raw sample data available – skipping individual "
              "runs plot.")
        return None

    n_qois = len(qois)
    fig, axes = plt.subplots(1, n_qois, figsize=(6 * n_qois, 5))
    if n_qois == 1:
        axes = [axes]

    for ax, qoi in zip(axes, qois):
        try:
            samples = np.squeeze(np.array(raw[qoi]))
            run_ids = np.arange(1, len(samples) + 1)

            ax.scatter(run_ids, samples, s=20, color=_PRIMARY_COLOR,
                       edgecolors="black", linewidths=0.4, alpha=0.85,
                       label="Individual runs")

            mean_val = float(np.squeeze(results.describe(qoi, "mean")))
            ax.axhline(mean_val, color=_SECONDARY_COLOR, linewidth=1.5,
                       linestyle="--", label=f"Mean = {mean_val:.4e}")

            ax.set_xlabel("Run #", fontsize=11)
            unit = _unit_for_qoi(qoi)
            ylabel = f"{_label_for_qoi(qoi)} [{unit}]" if unit else _label_for_qoi(qoi)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"Individual Runs – {_label_for_qoi(qoi)}",
                         fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
        except (KeyError, TypeError):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="grey")
            ax.set_title(_label_for_qoi(qoi))

    fig.tight_layout()
    filepath = os.path.join(output_dir, "qoi_individual_runs.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  ✓ QoI individual runs plot saved: {filepath}")
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

    # 4. Second-order Sobol indices heatmap
    saved_files.extend(
        plot_sobol_second_order_heatmap(results, qois, distributions,
                                       output_dir))

    # 5. First-order Sobol pie charts
    saved_files.extend(
        plot_sobol_first_order_pie(results, qois, distributions, output_dir))

    # 6. QoI statistics table (Mean & Std)
    saved_files.append(
        plot_qoi_statistics_table(results, qois, output_dir))

    # 7. Relative standard deviation bar plot (log-scale)
    saved_files.append(plot_relative_std(results, qois, output_dir))

    # 8. Input uncertainty PDFs
    pdf_path = plot_input_uncertainty_pdfs(distributions, output_dir)
    if pdf_path is not None:
        saved_files.append(pdf_path)

    # 9. QoI PDFs from PCE surrogate
    qoi_pdf_path = plot_qoi_pdf_pce(results, qois, output_dir)
    if qoi_pdf_path is not None:
        saved_files.append(qoi_pdf_path)

    # 10. Individual run values per QoI
    runs_path = plot_qoi_individual_runs(results, qois, output_dir)
    if runs_path is not None:
        saved_files.append(runs_path)

    print(f" >> Visualisation complete. {len(saved_files)} plot(s) generated.")
    return saved_files
