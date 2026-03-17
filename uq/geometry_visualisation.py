"""
geometry_visualisation.py – Schematic geometry plots for the OpenMC TBM model.

Generates annotated cross-section diagrams of the experiment geometry
directly from the YAML configuration, without requiring OpenMC.

Plots
-----
1. **Side view (Y-Z)** – Full room with target wheel layers and TBM.
2. **Top view (X-Z)** – Target wheel disk and TBM footprint at TBM height.
3. **Target layer detail** – Zoomed Z-axis stack-up with thicknesses.
4. **TBM detail (Y-Z)** – Zoomed TBM casing and ceramic fill.

Usage
-----
    from uq.geometry_visualisation import plot_geometry
    paths = plot_geometry("config/model_config.yaml", output_dir="plots")
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Material colour palette ──────────────────────────────────────────────────
_MAT_COLORS = {
    "Li":       "#E8C944",  # golden yellow
    "Cu":       "#CD7F32",  # copper/bronze
    "Water":    "#5DADE2",  # blue
    "Vacuum":   "#F0F0F0",  # very light grey
    "Graphite": "#555555",  # dark grey
    "Ti":       "#A0A0C0",  # steel blue-grey
    "Air":      "#E8F8E8",  # very faint green
    "Eurofer":  "#B0B0B0",  # medium grey
    "Li2TiO3":  "#E07050",  # red-orange (ceramic)
    "Room":     "#FAFAFA",  # near-white
}


# ---------------------------------------------------------------------------
# Geometry extraction from config
# ---------------------------------------------------------------------------

def _get_mean(value):
    """Return the mean/nominal value from a scalar or {mean: ...} dict."""
    if isinstance(value, dict):
        return value.get("mean", 0.0)
    return value


def compute_geometry(config):
    """
    Compute all geometry coordinates from a model configuration dict.

    Parameters
    ----------
    config : dict
        Parsed YAML model configuration (as returned by ``yaml.safe_load``).

    Returns
    -------
    dict
        Dictionary of geometry coordinates and dimensions.
    """
    geom = config.get("geometry", {})

    # Target layer thicknesses (cm)
    li_t       = float(geom.get("li_thickness",       0.02))
    cu_t       = float(geom.get("cu_thickness",       0.3))
    water_t    = float(geom.get("water_thickness",    0.6))
    vac1_t     = float(geom.get("vacuum_thickness_1", 1.5))
    graphite_t = float(_get_mean(geom.get("graphite_thickness", 0.7)))
    vac2_t     = float(geom.get("vacuum_thickness_2", 0.48))
    ti_t       = float(geom.get("ti_thickness",       0.6))
    air_gap    = float(geom.get("air_gap",            0.1))
    wh_r       = float(geom.get("wh_r",               50.0))

    # TBM
    eurofer_t      = float(geom.get("eurofer_thickness", 0.5))
    tbm_width      = float(geom.get("tbm_width",        7.0))
    tbm_thickness  = float(geom.get("tbm_thickness",    2.0))
    tbm_height     = float(geom.get("tbm_height",       3.0))
    tbm_pos_y      = float(geom.get("tbm_position_y",   -42.0))

    # Optional graphite liner
    graphite_liner_t = float(geom.get("graphite_liner_thickness", 0.0))

    # Z-coordinates of target layers
    z_li_lo    = -li_t / 2
    z_li_hi    =  li_t / 2
    z_cu1_hi   = z_li_hi + cu_t
    z_water_hi = z_cu1_hi + water_t
    z_cu2_hi   = z_water_hi + cu_t
    z_vac1_hi  = z_cu2_hi + vac1_t
    z_graph_hi = z_vac1_hi + graphite_t
    z_vac2_hi  = z_graph_hi + vac2_t
    z_ti_hi    = z_vac2_hi + ti_t

    # TBM coordinates
    base_z = z_ti_hi + air_gap
    tbm_z_start = base_z
    tbm_z_end   = base_z + tbm_thickness

    casing_x_left   = -(tbm_width / 2 + eurofer_t)
    casing_x_right  =  (tbm_width / 2 + eurofer_t)
    casing_y_bottom = tbm_pos_y - (tbm_height / 2 + eurofer_t)
    casing_y_top    = tbm_pos_y + (tbm_height / 2 + eurofer_t)

    inner_z_start = base_z + eurofer_t
    inner_z_end   = base_z + tbm_thickness - eurofer_t
    inner_x_left  = -tbm_width / 2
    inner_x_right =  tbm_width / 2
    inner_y_bottom = tbm_pos_y - tbm_height / 2
    inner_y_top    = tbm_pos_y + tbm_height / 2

    # If graphite liner, the ceramic stops earlier
    if graphite_liner_t > 0:
        liner_z_start = inner_z_end - graphite_liner_t
    else:
        liner_z_start = None

    # Room boundaries
    room = {
        "z_min": -50.0, "z_max": 250.0,
        "x_min": -150.0, "x_max": 150.0,
        "y_min": -100.0, "y_max": 100.0,
    }

    # Target layer boundaries (list of (z_lo, z_hi, name, thickness))
    layers = [
        (z_li_lo,    z_li_hi,    "Li",       li_t),
        (z_li_hi,    z_cu1_hi,   "Cu",       cu_t),
        (z_cu1_hi,   z_water_hi, "Water",    water_t),
        (z_water_hi, z_cu2_hi,   "Cu",       cu_t),
        (z_cu2_hi,   z_vac1_hi,  "Vacuum",   vac1_t),
        (z_vac1_hi,  z_graph_hi, "Graphite", graphite_t),
        (z_graph_hi, z_vac2_hi,  "Vacuum",   vac2_t),
        (z_vac2_hi,  z_ti_hi,    "Ti",       ti_t),
    ]

    return {
        "wh_r": wh_r,
        "layers": layers,
        "z_li_lo": z_li_lo,
        "z_ti_hi": z_ti_hi,
        "air_gap": air_gap,
        # TBM outer casing
        "tbm_z_start": tbm_z_start, "tbm_z_end": tbm_z_end,
        "casing_x_left": casing_x_left, "casing_x_right": casing_x_right,
        "casing_y_bottom": casing_y_bottom, "casing_y_top": casing_y_top,
        # TBM inner ceramic
        "inner_z_start": inner_z_start, "inner_z_end": inner_z_end,
        "inner_x_left": inner_x_left, "inner_x_right": inner_x_right,
        "inner_y_bottom": inner_y_bottom, "inner_y_top": inner_y_top,
        # TBM dimensions
        "eurofer_t": eurofer_t, "tbm_width": tbm_width,
        "tbm_thickness": tbm_thickness, "tbm_height": tbm_height,
        "tbm_pos_y": tbm_pos_y,
        # Graphite liner
        "graphite_liner_t": graphite_liner_t,
        "liner_z_start": liner_z_start,
        # Room
        "room": room,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _add_dimension_annotation(ax, xy_start, xy_end, text, offset=0,
                              color="black", fontsize=7, horizontal=True):
    """Draw a double-headed arrow between two points with a centred label."""
    x0, y0 = xy_start
    x1, y1 = xy_end
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.8),
    )
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if horizontal:
        ax.text(mx, my + offset, text, ha="center", va="bottom",
                fontsize=fontsize, color=color)
    else:
        ax.text(mx + offset, my, text, ha="left", va="center",
                fontsize=fontsize, color=color, rotation=90)


# ---------------------------------------------------------------------------
# 1. Side view (Y-Z cross section at x = 0)
# ---------------------------------------------------------------------------

def plot_side_view(config, output_dir):
    """
    Side-view cross section (Y-Z plane at x = 0) of the full room.

    Shows the target wheel as a vertical band at y ∈ [-R, +R], the
    layered target structure along Z, and the TBM box at its y-offset.

    Parameters
    ----------
    config : dict
        Parsed model configuration.
    output_dir : str
        Directory where the figure is saved.

    Returns
    -------
    str
        Path to the saved figure.
    """
    g = compute_geometry(config)
    room = g["room"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Side View (Y-Z cross section at x = 0)", fontsize=12,
                 fontweight="bold")

    # Room background
    room_w = room["z_max"] - room["z_min"]
    room_h = room["y_max"] - room["y_min"]
    ax.add_patch(plt.Rectangle(
        (room["z_min"], room["y_min"]), room_w, room_h,
        fc=_MAT_COLORS["Room"], ec="black", lw=1.5, linestyle="--",
        label="Air (room)"))

    # Target wheel disk (side view: rectangle in Y-Z)
    wh_r = g["wh_r"]
    z_lo = g["z_li_lo"]
    z_hi = g["z_ti_hi"]
    wheel_w = z_hi - z_lo
    ax.add_patch(plt.Rectangle(
        (z_lo, -wh_r), wheel_w, 2 * wh_r,
        fc=_MAT_COLORS["Air"], ec="black", lw=0.8))

    # Target layers as horizontal bands
    for z0, z1, name, _thick in g["layers"]:
        color = _MAT_COLORS.get(name, "#CCCCCC")
        ax.add_patch(plt.Rectangle(
            (z0, -wh_r), z1 - z0, 2 * wh_r,
            fc=color, ec="grey", lw=0.3))

    # TBM casing (Eurofer)
    tbm_w = g["tbm_z_end"] - g["tbm_z_start"]
    tbm_h = g["casing_y_top"] - g["casing_y_bottom"]
    ax.add_patch(plt.Rectangle(
        (g["tbm_z_start"], g["casing_y_bottom"]), tbm_w, tbm_h,
        fc=_MAT_COLORS["Eurofer"], ec="black", lw=1.0))

    # TBM inner ceramic
    inner_w = g["inner_z_end"] - g["inner_z_start"]
    inner_h = g["inner_y_top"] - g["inner_y_bottom"]

    if g["graphite_liner_t"] > 0 and g["liner_z_start"] is not None:
        # Ceramic region (reduced)
        ceramic_w = g["liner_z_start"] - g["inner_z_start"]
        ax.add_patch(plt.Rectangle(
            (g["inner_z_start"], g["inner_y_bottom"]), ceramic_w, inner_h,
            fc=_MAT_COLORS["Li2TiO3"], ec="black", lw=0.5))
        # Graphite liner
        ax.add_patch(plt.Rectangle(
            (g["liner_z_start"], g["inner_y_bottom"]),
            g["graphite_liner_t"], inner_h,
            fc=_MAT_COLORS["Graphite"], ec="black", lw=0.5))
    else:
        ax.add_patch(plt.Rectangle(
            (g["inner_z_start"], g["inner_y_bottom"]), inner_w, inner_h,
            fc=_MAT_COLORS["Li2TiO3"], ec="black", lw=0.5))

    # Labels
    ax.text(g["z_li_lo"] - 1, 0, "Target\nWheel", ha="right", va="center",
            fontsize=8, fontstyle="italic")
    tbm_cz = (g["tbm_z_start"] + g["tbm_z_end"]) / 2
    tbm_cy = (g["casing_y_bottom"] + g["casing_y_top"]) / 2
    ax.text(tbm_cz, tbm_cy, "TBM", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#666666", alpha=0.7))

    # Wheel radius dimension
    _add_dimension_annotation(
        ax, (z_lo - 3, 0), (z_lo - 3, wh_r),
        f"R = {wh_r:.0f} cm", offset=1.5, fontsize=7, horizontal=True)

    # Axis setup
    ax.set_xlabel("Z (cm)", fontsize=10)
    ax.set_ylabel("Y (cm)", fontsize=10)
    ax.set_xlim(room["z_min"] - 5, room["z_max"] + 5)
    ax.set_ylim(room["y_min"] - 5, room["y_max"] + 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle=":")

    # Legend
    legend_patches = [
        mpatches.Patch(fc=_MAT_COLORS["Li"], ec="grey", label="Li (target)"),
        mpatches.Patch(fc=_MAT_COLORS["Cu"], ec="grey", label="Cu"),
        mpatches.Patch(fc=_MAT_COLORS["Water"], ec="grey", label="Water"),
        mpatches.Patch(fc=_MAT_COLORS["Graphite"], ec="grey", label="Graphite"),
        mpatches.Patch(fc=_MAT_COLORS["Ti"], ec="grey", label="Ti"),
        mpatches.Patch(fc=_MAT_COLORS["Eurofer"], ec="grey",
                       label="Eurofer-97 (casing)"),
        mpatches.Patch(fc=_MAT_COLORS["Li2TiO3"], ec="grey",
                       label=r"Li$_2$TiO$_3$ (ceramic)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
              framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, "geometry_side_view.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Top view (X-Z cross section at y = tbm_position_y)
# ---------------------------------------------------------------------------

def plot_top_view(config, output_dir):
    """
    Top-view cross section (X-Z plane at y = tbm_position_y).

    Shows the target wheel disk (circle cross-section) and the TBM box.

    Parameters
    ----------
    config : dict
        Parsed model configuration.
    output_dir : str
        Directory where the figure is saved.

    Returns
    -------
    str
        Path to the saved figure.
    """
    g = compute_geometry(config)
    room = g["room"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        f"Top View (X-Z cross section at y = {g['tbm_pos_y']:.1f} cm)",
        fontsize=12, fontweight="bold")

    # Room background
    room_w = room["z_max"] - room["z_min"]
    room_h = room["x_max"] - room["x_min"]
    ax.add_patch(plt.Rectangle(
        (room["z_min"], room["x_min"]), room_w, room_h,
        fc=_MAT_COLORS["Room"], ec="black", lw=1.5, linestyle="--"))

    # Target wheel layers (horizontal bands across full x-range within disc)
    # At y = tbm_pos_y, the wheel disc half-width in x is
    # sqrt(R² - y²) if |y| < R
    wh_r = g["wh_r"]
    y_slice = abs(g["tbm_pos_y"])
    if y_slice < wh_r:
        x_half = np.sqrt(wh_r**2 - y_slice**2)
        for z0, z1, name, _thick in g["layers"]:
            color = _MAT_COLORS.get(name, "#CCCCCC")
            ax.add_patch(plt.Rectangle(
                (z0, -x_half), z1 - z0, 2 * x_half,
                fc=color, ec="grey", lw=0.3))
        # Wheel outline
        ax.add_patch(plt.Rectangle(
            (g["z_li_lo"], -x_half),
            g["z_ti_hi"] - g["z_li_lo"], 2 * x_half,
            fc="none", ec="black", lw=0.8))
        ax.text(g["z_li_lo"] - 2, 0, "Target\nWheel", ha="right",
                va="center", fontsize=8, fontstyle="italic")
    else:
        # TBM is outside the wheel radius – just note it
        ax.text(0, 0, "(Target wheel not visible\nat this y-slice)",
                ha="center", va="center", fontsize=9, color="grey")

    # TBM casing
    tbm_z_w = g["tbm_z_end"] - g["tbm_z_start"]
    tbm_x_w = g["casing_x_right"] - g["casing_x_left"]
    ax.add_patch(plt.Rectangle(
        (g["tbm_z_start"], g["casing_x_left"]), tbm_z_w, tbm_x_w,
        fc=_MAT_COLORS["Eurofer"], ec="black", lw=1.0))

    # TBM inner ceramic
    inner_z_w = g["inner_z_end"] - g["inner_z_start"]
    inner_x_w = g["inner_x_right"] - g["inner_x_left"]

    if g["graphite_liner_t"] > 0 and g["liner_z_start"] is not None:
        ceramic_z_w = g["liner_z_start"] - g["inner_z_start"]
        ax.add_patch(plt.Rectangle(
            (g["inner_z_start"], g["inner_x_left"]), ceramic_z_w, inner_x_w,
            fc=_MAT_COLORS["Li2TiO3"], ec="black", lw=0.5))
        ax.add_patch(plt.Rectangle(
            (g["liner_z_start"], g["inner_x_left"]),
            g["graphite_liner_t"], inner_x_w,
            fc=_MAT_COLORS["Graphite"], ec="black", lw=0.5))
    else:
        ax.add_patch(plt.Rectangle(
            (g["inner_z_start"], g["inner_x_left"]), inner_z_w, inner_x_w,
            fc=_MAT_COLORS["Li2TiO3"], ec="black", lw=0.5))

    # TBM label
    tbm_cz = (g["tbm_z_start"] + g["tbm_z_end"]) / 2
    tbm_cx = (g["casing_x_left"] + g["casing_x_right"]) / 2
    ax.text(tbm_cz, tbm_cx, "TBM", ha="center", va="center", fontsize=9,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#666666", alpha=0.7))

    # TBM width annotation
    _add_dimension_annotation(
        ax, (g["tbm_z_end"] + 1, g["casing_x_left"]),
        (g["tbm_z_end"] + 1, g["casing_x_right"]),
        f"{tbm_x_w:.1f} cm", offset=0.3, fontsize=7, horizontal=True)

    ax.set_xlabel("Z (cm)", fontsize=10)
    ax.set_ylabel("X (cm)", fontsize=10)
    ax.set_xlim(room["z_min"] - 5, room["z_max"] + 5)
    ax.set_ylim(room["x_min"] - 5, room["x_max"] + 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle=":")

    # Legend
    legend_patches = [
        mpatches.Patch(fc=_MAT_COLORS["Li"], ec="grey", label="Li (target)"),
        mpatches.Patch(fc=_MAT_COLORS["Cu"], ec="grey", label="Cu"),
        mpatches.Patch(fc=_MAT_COLORS["Water"], ec="grey", label="Water"),
        mpatches.Patch(fc=_MAT_COLORS["Graphite"], ec="grey", label="Graphite"),
        mpatches.Patch(fc=_MAT_COLORS["Ti"], ec="grey", label="Ti"),
        mpatches.Patch(fc=_MAT_COLORS["Eurofer"], ec="grey",
                       label="Eurofer-97"),
        mpatches.Patch(fc=_MAT_COLORS["Li2TiO3"], ec="grey",
                       label=r"Li$_2$TiO$_3$"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
              framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, "geometry_top_view.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Target layer detail (zoomed Z-axis stack-up)
# ---------------------------------------------------------------------------

def plot_target_layers(config, output_dir):
    """
    Zoomed schematic of the target wheel layer stack-up along the Z axis.

    Each layer is drawn as a coloured horizontal band with its material
    label and thickness annotated.

    Parameters
    ----------
    config : dict
        Parsed model configuration.
    output_dir : str
        Directory where the figure is saved.

    Returns
    -------
    str
        Path to the saved figure.
    """
    g = compute_geometry(config)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Target Wheel Layer Stack-up (Z axis)", fontsize=12,
                 fontweight="bold")

    layers = g["layers"]
    bar_half_width = 10.0  # arbitrary visual half-width in y

    for z0, z1, name, thick in layers:
        color = _MAT_COLORS.get(name, "#CCCCCC")
        ax.add_patch(plt.Rectangle(
            (-bar_half_width, z0), 2 * bar_half_width, z1 - z0,
            fc=color, ec="black", lw=0.6))
        # Material label centred in the layer
        mid_z = (z0 + z1) / 2
        ax.text(0, mid_z, f"{name}\n({thick:.2f} cm)",
                ha="center", va="center", fontsize=8, fontweight="bold")

    # Dimension arrows on the right
    arrow_x = bar_half_width + 2
    for z0, z1, name, thick in layers:
        if thick >= 0.1:  # Only annotate layers thick enough to read
            _add_dimension_annotation(
                ax, (arrow_x, z0), (arrow_x, z1),
                f"{thick:.2f} cm", offset=0.5, fontsize=6,
                horizontal=True)

    # Total height
    z_min = layers[0][0]
    z_max = layers[-1][1]
    total = z_max - z_min
    _add_dimension_annotation(
        ax, (arrow_x + 5, z_min), (arrow_x + 5, z_max),
        f"Total: {total:.2f} cm", offset=0.3, fontsize=7,
        horizontal=True)

    # Air gap + TBM start indicator
    air_gap = g["air_gap"]
    tbm_z_start = g["tbm_z_start"]
    ax.axhline(y=z_max, color="grey", linestyle=":", lw=0.5)
    ax.axhline(y=tbm_z_start, color="grey", linestyle=":", lw=0.5)
    ax.text(bar_half_width + 1, (z_max + tbm_z_start) / 2,
            f"Air gap\n({air_gap:.2f} cm)", ha="left", va="center",
            fontsize=7, color="grey")

    # Neutron source indicator
    z_li_lo = g["z_li_lo"]
    z_li_hi = layers[0][1]
    ax.annotate("14.1 MeV\nneutron source", xy=(-bar_half_width - 1, (z_li_lo + z_li_hi) / 2),
                xytext=(-bar_half_width - 6, (z_li_lo + z_li_hi) / 2 - 1),
                fontsize=7, color="#CC0000",
                arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.0),
                ha="center", va="top")

    # Axis
    z_pad = max(0.5, total * 0.15)
    ax.set_xlim(-bar_half_width - 8, arrow_x + 10)
    ax.set_ylim(z_min - z_pad, tbm_z_start + z_pad)
    ax.set_ylabel("Z (cm)", fontsize=10)
    ax.set_xlabel("(schematic width)", fontsize=9, color="grey")
    ax.tick_params(axis="x", labelbottom=False)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Legend
    seen = set()
    legend_patches = []
    for _, _, name, _ in layers:
        if name not in seen and name != "Vacuum":
            seen.add(name)
            legend_patches.append(
                mpatches.Patch(fc=_MAT_COLORS.get(name, "#CCC"),
                               ec="grey", label=name))
    ax.legend(handles=legend_patches, fontsize=7, loc="upper left",
              framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, "geometry_target_layers.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. TBM detail (Y-Z cross section, zoomed)
# ---------------------------------------------------------------------------

def plot_tbm_detail(config, output_dir):
    """
    Zoomed cross section of the Test Blanket Module (Y-Z plane).

    Shows the Eurofer-97 casing, the Li₂TiO₃ ceramic fill, and
    (if present) the graphite liner layer.  All dimensions are annotated.

    Parameters
    ----------
    config : dict
        Parsed model configuration.
    output_dir : str
        Directory where the figure is saved.

    Returns
    -------
    str
        Path to the saved figure.
    """
    g = compute_geometry(config)

    fig, ax = plt.subplots(figsize=(8, 6))
    title = "TBM Cross Section (Y-Z at x = 0)"
    if g["graphite_liner_t"] > 0:
        title += " — with graphite liner"
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Casing (Eurofer)
    casing_z = g["tbm_z_start"]
    casing_dz = g["tbm_z_end"] - g["tbm_z_start"]
    casing_y = g["casing_y_bottom"]
    casing_dy = g["casing_y_top"] - g["casing_y_bottom"]
    ax.add_patch(plt.Rectangle(
        (casing_z, casing_y), casing_dz, casing_dy,
        fc=_MAT_COLORS["Eurofer"], ec="black", lw=1.2,
        label="Eurofer-97"))

    # Inner ceramic
    inner_z = g["inner_z_start"]
    inner_dz = g["inner_z_end"] - g["inner_z_start"]
    inner_y = g["inner_y_bottom"]
    inner_dy = g["inner_y_top"] - g["inner_y_bottom"]

    if g["graphite_liner_t"] > 0 and g["liner_z_start"] is not None:
        ceramic_dz = g["liner_z_start"] - inner_z
        ax.add_patch(plt.Rectangle(
            (inner_z, inner_y), ceramic_dz, inner_dy,
            fc=_MAT_COLORS["Li2TiO3"], ec="black", lw=0.6,
            label=r"Li$_2$TiO$_3$ ceramic"))
        ax.add_patch(plt.Rectangle(
            (g["liner_z_start"], inner_y),
            g["graphite_liner_t"], inner_dy,
            fc=_MAT_COLORS["Graphite"], ec="black", lw=0.6,
            label="Graphite liner"))
    else:
        ax.add_patch(plt.Rectangle(
            (inner_z, inner_y), inner_dz, inner_dy,
            fc=_MAT_COLORS["Li2TiO3"], ec="black", lw=0.6,
            label=r"Li$_2$TiO$_3$ ceramic"))

    # Dimension annotations
    eurofer_t = g["eurofer_t"]
    # Eurofer thickness (bottom wall)
    _add_dimension_annotation(
        ax,
        (casing_z - 0.3, casing_y),
        (casing_z - 0.3, inner_y),
        f"{eurofer_t:.1f} cm", offset=-0.4, fontsize=7, horizontal=True)

    # TBM total thickness (z)
    _add_dimension_annotation(
        ax,
        (casing_z, g["casing_y_top"] + 0.3),
        (g["tbm_z_end"], g["casing_y_top"] + 0.3),
        f"{g['tbm_thickness']:.1f} cm",
        offset=0.15, fontsize=7, horizontal=True)

    # TBM total height (y)
    _add_dimension_annotation(
        ax,
        (g["tbm_z_end"] + 0.3, casing_y),
        (g["tbm_z_end"] + 0.3, g["casing_y_top"]),
        f"{g['tbm_height'] + 2 * eurofer_t:.1f} cm",
        offset=0.15, fontsize=7, horizontal=True)

    # Graphite liner thickness if present
    if g["graphite_liner_t"] > 0 and g["liner_z_start"] is not None:
        mid_y = (inner_y + g["inner_y_top"]) / 2
        _add_dimension_annotation(
            ax,
            (g["liner_z_start"], mid_y - 0.6),
            (g["inner_z_end"], mid_y - 0.6),
            f"Liner: {g['graphite_liner_t']:.1f} cm",
            offset=0.1, fontsize=6, horizontal=True)

    # Centre label
    tbm_cz = (g["tbm_z_start"] + g["tbm_z_end"]) / 2
    tbm_cy = (g["casing_y_bottom"] + g["casing_y_top"]) / 2
    if g["graphite_liner_t"] > 0 and g["liner_z_start"] is not None:
        ceramic_cz = (inner_z + g["liner_z_start"]) / 2
    else:
        ceramic_cz = (inner_z + g["inner_z_end"]) / 2
    ax.text(ceramic_cz, tbm_cy, r"Li$_2$TiO$_3$" + "\n(ceramic)",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#D05030", alpha=0.8))

    # Target direction arrow
    ax.annotate(
        "← Target wheel", xy=(casing_z - 0.1, tbm_cy),
        xytext=(casing_z - 1.5, tbm_cy), fontsize=7, color="grey",
        arrowprops=dict(arrowstyle="<-", color="grey", lw=0.8),
        ha="right", va="center")

    # Axis
    pad_z = max(0.5, casing_dz * 0.4)
    pad_y = max(0.5, casing_dy * 0.2)
    ax.set_xlim(casing_z - pad_z - 1.5, g["tbm_z_end"] + pad_z + 0.5)
    ax.set_ylim(casing_y - pad_y, g["casing_y_top"] + pad_y + 0.5)
    ax.set_xlabel("Z (cm)", fontsize=10)
    ax.set_ylabel("Y (cm)", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, "geometry_tbm_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Full room overview with annotations
# ---------------------------------------------------------------------------

def plot_room_overview(config, output_dir):
    """
    Annotated overview of the full experimental room (Y-Z plane).

    Shows room boundaries with their coordinates, the target wheel,
    TBM, and key distances (beam-axis to TBM, room dimensions).

    Parameters
    ----------
    config : dict
        Parsed model configuration.
    output_dir : str
        Directory where the figure is saved.

    Returns
    -------
    str
        Path to the saved figure.
    """
    g = compute_geometry(config)
    room = g["room"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Experiment Room Overview (Y-Z plane)", fontsize=13,
                 fontweight="bold")

    # Room
    room_w = room["z_max"] - room["z_min"]
    room_h = room["y_max"] - room["y_min"]
    ax.add_patch(plt.Rectangle(
        (room["z_min"], room["y_min"]), room_w, room_h,
        fc=_MAT_COLORS["Room"], ec="black", lw=2.0, linestyle="-"))

    # Room dimension labels
    ax.text(room["z_min"], room["y_min"] - 4,
            f'z = {room["z_min"]:.0f} cm', ha="center", fontsize=7,
            color="dimgrey")
    ax.text(room["z_max"], room["y_min"] - 4,
            f'z = {room["z_max"]:.0f} cm', ha="center", fontsize=7,
            color="dimgrey")
    ax.text(room["z_min"] - 4, room["y_min"],
            f'y = {room["y_min"]:.0f}', ha="right", fontsize=7,
            color="dimgrey", rotation=90)
    ax.text(room["z_min"] - 4, room["y_max"],
            f'y = {room["y_max"]:.0f}', ha="right", fontsize=7,
            color="dimgrey", rotation=90)

    # Room dimensions
    _add_dimension_annotation(
        ax, (room["z_min"], room["y_max"] + 3),
        (room["z_max"], room["y_max"] + 3),
        f'{room_w:.0f} cm', offset=1.5, fontsize=8, color="dimgrey")
    _add_dimension_annotation(
        ax, (room["z_max"] + 3, room["y_min"]),
        (room["z_max"] + 3, room["y_max"]),
        f'{room_h:.0f} cm', offset=2, fontsize=8, color="dimgrey",
        horizontal=False)

    # Target wheel (simplified: rectangle in Y-Z)
    wh_r = g["wh_r"]
    z_lo = g["z_li_lo"]
    z_hi = g["z_ti_hi"]
    wheel_w = z_hi - z_lo
    ax.add_patch(plt.Rectangle(
        (z_lo, -wh_r), wheel_w, 2 * wh_r,
        fc="#FFFDE0", ec="black", lw=1.0))

    # Simplified target layers (merged for readability)
    for z0, z1, name, _thick in g["layers"]:
        color = _MAT_COLORS.get(name, "#CCCCCC")
        ax.add_patch(plt.Rectangle(
            (z0, -wh_r), z1 - z0, 2 * wh_r,
            fc=color, ec="grey", lw=0.2, alpha=0.8))
    ax.text((z_lo + z_hi) / 2, 0, "Target\nWheel", ha="center",
            va="center", fontsize=9, fontweight="bold")

    # Beam axis
    ax.axhline(y=0, color="red", linestyle="-.", lw=0.7, alpha=0.6)
    ax.text(room["z_max"] - 10, 2, "beam axis (y = 0)",
            fontsize=7, color="red", ha="right")

    # TBM
    tbm_w = g["tbm_z_end"] - g["tbm_z_start"]
    tbm_h = g["casing_y_top"] - g["casing_y_bottom"]
    ax.add_patch(plt.Rectangle(
        (g["tbm_z_start"], g["casing_y_bottom"]), tbm_w, tbm_h,
        fc=_MAT_COLORS["Eurofer"], ec="black", lw=1.2))
    # Inner ceramic
    inner_w = g["inner_z_end"] - g["inner_z_start"]
    inner_h = g["inner_y_top"] - g["inner_y_bottom"]
    ax.add_patch(plt.Rectangle(
        (g["inner_z_start"], g["inner_y_bottom"]), inner_w, inner_h,
        fc=_MAT_COLORS["Li2TiO3"], ec="black", lw=0.5))
    tbm_cz = (g["tbm_z_start"] + g["tbm_z_end"]) / 2
    tbm_cy = (g["casing_y_bottom"] + g["casing_y_top"]) / 2
    ax.text(tbm_cz, tbm_cy, "TBM", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#666666", alpha=0.8))

    # Distance from beam axis to TBM
    _add_dimension_annotation(
        ax, (g["tbm_z_end"] + 3, 0),
        (g["tbm_z_end"] + 3, g["tbm_pos_y"]),
        f'{abs(g["tbm_pos_y"]):.1f} cm', offset=1.5, fontsize=7,
        color="#0066CC")

    # Vacuum boundary label
    ax.text((room["z_min"] + room["z_max"]) / 2, room["y_max"] - 3,
            "Vacuum boundary conditions", ha="center", fontsize=8,
            color="dimgrey", fontstyle="italic")

    ax.set_xlabel("Z (cm)", fontsize=10)
    ax.set_ylabel("Y (cm)", fontsize=10)
    ax.set_xlim(room["z_min"] - 12, room["z_max"] + 12)
    ax.set_ylim(room["y_min"] - 10, room["y_max"] + 10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linestyle=":")

    # Legend
    legend_patches = [
        mpatches.Patch(fc=_MAT_COLORS["Li"], ec="grey", label="Li (target)"),
        mpatches.Patch(fc=_MAT_COLORS["Cu"], ec="grey", label="Cu"),
        mpatches.Patch(fc=_MAT_COLORS["Water"], ec="grey", label="Water"),
        mpatches.Patch(fc=_MAT_COLORS["Graphite"], ec="grey", label="Graphite"),
        mpatches.Patch(fc=_MAT_COLORS["Ti"], ec="grey", label="Ti"),
        mpatches.Patch(fc=_MAT_COLORS["Eurofer"], ec="grey",
                       label="Eurofer-97 (TBM casing)"),
        mpatches.Patch(fc=_MAT_COLORS["Li2TiO3"], ec="grey",
                       label=r"Li$_2$TiO$_3$ (ceramic)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper left",
              framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(output_dir, "geometry_room_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_geometry(config_or_path, output_dir=None):
    """
    Generate all geometry plots from a model configuration.

    Parameters
    ----------
    config_or_path : dict or str
        Either a parsed configuration dict or a path to a YAML file.
    output_dir : str, optional
        Directory for saved figures.  Defaults to ``geometry_plots/``.

    Returns
    -------
    list[str]
        Paths to the saved figures.
    """
    import yaml

    if isinstance(config_or_path, str):
        with open(config_or_path, "r") as fh:
            config = yaml.safe_load(fh)
    else:
        config = config_or_path

    if output_dir is None:
        output_dir = "geometry_plots"
    os.makedirs(output_dir, exist_ok=True)

    paths = [
        plot_side_view(config, output_dir),
        plot_top_view(config, output_dir),
        plot_target_layers(config, output_dir),
        plot_tbm_detail(config, output_dir),
        plot_room_overview(config, output_dir),
    ]
    return paths
