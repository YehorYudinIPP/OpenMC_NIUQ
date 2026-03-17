"""
Tests for uq.geometry_visualisation – schematic geometry plotting.

Validates that geometry is computed correctly from the YAML config and that
all five plot functions produce non-empty PNG files.  Does not require OpenMC.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import yaml

import sys

_uq_dir = os.path.join(os.path.dirname(__file__), os.pardir, "uq")
if _uq_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_uq_dir))

from geometry_visualisation import (
    _get_mean,
    compute_geometry,
    plot_geometry,
    plot_room_overview,
    plot_side_view,
    plot_target_layers,
    plot_tbm_detail,
    plot_top_view,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "uq", "config", "model_config.yaml")
_CONFIG_LINER_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "uq", "config",
    "model_config_graphite_liner.yaml")


@pytest.fixture()
def config():
    with open(_CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


@pytest.fixture()
def config_liner():
    with open(_CONFIG_LINER_PATH) as fh:
        return yaml.safe_load(fh)


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Tests – _get_mean helper
# ---------------------------------------------------------------------------

class TestGetMean:
    def test_scalar(self):
        assert _get_mean(0.7) == 0.7

    def test_dict(self):
        assert _get_mean({"mean": 3.43, "relative_stdev": 0.02}) == 3.43

    def test_dict_missing_mean(self):
        assert _get_mean({"relative_stdev": 0.05}) == 0.0


# ---------------------------------------------------------------------------
# Tests – compute_geometry
# ---------------------------------------------------------------------------

class TestComputeGeometry:
    """Verify that geometry coordinates match the model code exactly."""

    def test_target_layer_count(self, config):
        g = compute_geometry(config)
        assert len(g["layers"]) == 8

    def test_target_layer_z_order(self, config):
        g = compute_geometry(config)
        for z0, z1, _name, _t in g["layers"]:
            assert z1 > z0, f"Layer z-bounds out of order: {z0} >= {z1}"

    def test_target_layer_contiguous(self, config):
        """Each layer's start must equal the previous layer's end."""
        g = compute_geometry(config)
        layers = g["layers"]
        for i in range(1, len(layers)):
            assert layers[i][0] == pytest.approx(layers[i - 1][1], abs=1e-10)

    def test_li_layer_symmetric(self, config):
        g = compute_geometry(config)
        z0, z1 = g["layers"][0][0], g["layers"][0][1]
        assert z0 == pytest.approx(-z1)

    def test_total_target_thickness(self, config):
        """Sum of all layer thicknesses should equal z_ti_hi - z_li_lo."""
        g = compute_geometry(config)
        total = sum(t for _, _, _, t in g["layers"])
        assert total == pytest.approx(g["z_ti_hi"] - g["z_li_lo"])

    def test_default_layer_thicknesses(self, config):
        g = compute_geometry(config)
        expected = [
            ("Li",       0.02),
            ("Cu",       0.30),
            ("Water",    0.60),
            ("Cu",       0.30),
            ("Vacuum",   1.50),
            ("Graphite", 0.70),
            ("Vacuum",   0.48),
            ("Ti",       0.60),
        ]
        for (_, _, name, t), (exp_name, exp_t) in zip(g["layers"], expected):
            assert name == exp_name
            assert t == pytest.approx(exp_t)

    def test_tbm_z_offset(self, config):
        """TBM starts at z_ti_hi + air_gap."""
        g = compute_geometry(config)
        assert g["tbm_z_start"] == pytest.approx(g["z_ti_hi"] + g["air_gap"])

    def test_tbm_z_extent(self, config):
        g = compute_geometry(config)
        assert g["tbm_z_end"] - g["tbm_z_start"] == pytest.approx(
            g["tbm_thickness"])

    def test_tbm_casing_surrounds_inner(self, config):
        g = compute_geometry(config)
        assert g["casing_x_left"] < g["inner_x_left"]
        assert g["casing_x_right"] > g["inner_x_right"]
        assert g["casing_y_bottom"] < g["inner_y_bottom"]
        assert g["casing_y_top"] > g["inner_y_top"]
        assert g["tbm_z_start"] < g["inner_z_start"]
        assert g["tbm_z_end"] > g["inner_z_end"]

    def test_eurofer_gap_consistent(self, config):
        """Gap between casing and inner must equal eurofer thickness."""
        g = compute_geometry(config)
        e = g["eurofer_t"]
        assert g["inner_z_start"] - g["tbm_z_start"] == pytest.approx(e)
        assert g["tbm_z_end"] - g["inner_z_end"] == pytest.approx(e)
        assert g["inner_x_left"] - g["casing_x_left"] == pytest.approx(e)
        assert g["casing_x_right"] - g["inner_x_right"] == pytest.approx(e)

    def test_room_boundaries(self, config):
        g = compute_geometry(config)
        room = g["room"]
        assert room["z_min"] == -50.0
        assert room["z_max"] == 250.0
        assert room["x_min"] == -150.0
        assert room["x_max"] == 150.0
        assert room["y_min"] == -100.0
        assert room["y_max"] == 100.0

    def test_no_graphite_liner_by_default(self, config):
        g = compute_geometry(config)
        assert g["graphite_liner_t"] == 0.0
        assert g["liner_z_start"] is None


class TestComputeGeometryLiner:
    """Tests specific to the graphite-liner variant."""

    def test_has_graphite_liner(self, config_liner):
        g = compute_geometry(config_liner)
        assert g["graphite_liner_t"] == pytest.approx(0.5)
        assert g["liner_z_start"] is not None

    def test_liner_inside_inner(self, config_liner):
        g = compute_geometry(config_liner)
        assert g["liner_z_start"] >= g["inner_z_start"]
        assert g["liner_z_start"] + g["graphite_liner_t"] == pytest.approx(
            g["inner_z_end"])


# ---------------------------------------------------------------------------
# Tests – Plot functions
# ---------------------------------------------------------------------------

class TestPlotSideView:
    def test_creates_file(self, config, tmp_dir):
        path = plot_side_view(config, tmp_dir)
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_file_nonempty(self, config, tmp_dir):
        path = plot_side_view(config, tmp_dir)
        assert os.path.getsize(path) > 0


class TestPlotTopView:
    def test_creates_file(self, config, tmp_dir):
        path = plot_top_view(config, tmp_dir)
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_file_nonempty(self, config, tmp_dir):
        path = plot_top_view(config, tmp_dir)
        assert os.path.getsize(path) > 0


class TestPlotTargetLayers:
    def test_creates_file(self, config, tmp_dir):
        path = plot_target_layers(config, tmp_dir)
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_file_nonempty(self, config, tmp_dir):
        path = plot_target_layers(config, tmp_dir)
        assert os.path.getsize(path) > 0


class TestPlotTbmDetail:
    def test_creates_file(self, config, tmp_dir):
        path = plot_tbm_detail(config, tmp_dir)
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_file_nonempty(self, config, tmp_dir):
        path = plot_tbm_detail(config, tmp_dir)
        assert os.path.getsize(path) > 0

    def test_graphite_liner_variant(self, config_liner, tmp_dir):
        path = plot_tbm_detail(config_liner, tmp_dir)
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestPlotRoomOverview:
    def test_creates_file(self, config, tmp_dir):
        path = plot_room_overview(config, tmp_dir)
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_file_nonempty(self, config, tmp_dir):
        path = plot_room_overview(config, tmp_dir)
        assert os.path.getsize(path) > 0


class TestPlotGeometry:
    """Tests for the top-level plot_geometry() function."""

    def test_creates_five_files(self, config, tmp_dir):
        paths = plot_geometry(config, output_dir=tmp_dir)
        assert len(paths) == 5
        for p in paths:
            assert os.path.isfile(p)
            assert p.endswith(".png")
            assert os.path.getsize(p) > 0

    def test_accepts_yaml_path(self, tmp_dir):
        paths = plot_geometry(_CONFIG_PATH, output_dir=tmp_dir)
        assert len(paths) == 5

    def test_default_output_dir(self, config):
        out = os.path.join(tempfile.mkdtemp(), "geometry_plots")
        try:
            paths = plot_geometry(config, output_dir=out)
            assert os.path.isdir(out)
            assert len(paths) == 5
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_graphite_liner_config(self, tmp_dir):
        paths = plot_geometry(_CONFIG_LINER_PATH, output_dir=tmp_dir)
        assert len(paths) == 5
        for p in paths:
            assert os.path.isfile(p)
