"""
Tests for the monoblock variant of the OpenMC TBM model configuration.

Validates that ``model_config_monoblock.yaml`` loads correctly, sets
``pebbles_or_monoblock`` to ``'monoblock'``, and preserves base geometry
and material parameters.

Also validates that all target-layer and TBM geometry parameters survive
being declared as uncertain (dict with 'mean' key) via ``_get_mean()``.

These tests do not require OpenMC or EasyVVUQ; they exercise the pure-Python
configuration logic.
"""

import os
import sys

import pytest
import yaml

# ── Importable whether pytest is run from repo root or from tests/ ────────
_uq_dir = os.path.join(os.path.dirname(__file__), os.pardir, "uq")
if _uq_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_uq_dir))

from openmc_model_run import _get_mean, load_config


# ---------------------------------------------------------------------------
# Monoblock config file
# ---------------------------------------------------------------------------

class TestMonoblockConfig:
    """Tests for the monoblock YAML configuration file."""

    @pytest.fixture()
    def config_path(self):
        return os.path.join(
            os.path.dirname(__file__), os.pardir,
            "uq", "config", "model_config_monoblock.yaml",
        )

    def test_config_file_exists(self, config_path):
        assert os.path.isfile(config_path)

    def test_config_loads_successfully(self, config_path):
        config = load_config(config_path)
        assert config is not None

    def test_config_selects_monoblock(self, config_path):
        config = load_config(config_path)
        geom = config.get("geometry", {})
        assert geom.get("pebbles_or_monoblock") == "monoblock"

    def test_config_has_no_pebble_params(self, config_path):
        """Monoblock config should not have pebble bed parameters."""
        config = load_config(config_path)
        geom = config.get("geometry", {})
        assert "pebble_radius" not in geom
        assert "packing_fraction" not in geom

    def test_config_preserves_base_geometry(self, config_path):
        """All base geometry parameters should still be present."""
        config = load_config(config_path)
        geom = config["geometry"]
        assert float(geom["eurofer_thickness"]) == pytest.approx(0.5)
        assert float(geom["tbm_width"]) == pytest.approx(7.0)
        assert float(geom["tbm_thickness"]) == pytest.approx(2.0)
        assert float(geom["tbm_height"]) == pytest.approx(3.0)

    def test_config_preserves_base_materials(self, config_path):
        """Materials section should be unchanged from the base model."""
        config = load_config(config_path)
        li_cer = config["materials"]["li_ceramic"]
        assert float(li_cer["density"]["mean"]) == pytest.approx(3.43)

    def test_config_has_target_layer_thicknesses(self, config_path):
        config = load_config(config_path)
        geom = config["geometry"]
        assert "li_thickness" in geom
        assert "cu_thickness" in geom
        assert "water_thickness" in geom
        assert "graphite_thickness" in geom
        assert "ti_thickness" in geom

    def test_config_qoi_list(self, config_path):
        config = load_config(config_path)
        qois = config["output"]["qoi"]
        assert "tritium_production_rate" in qois
        assert "total_neutron_flux" in qois


# ---------------------------------------------------------------------------
# _get_mean robustness: all target/geometry parameters handle uncertain dicts
# ---------------------------------------------------------------------------

class TestGetMeanRobustness:
    """Verify _get_mean handles uncertain dicts for every target/geometry param."""

    @pytest.mark.parametrize("param_name,default", [
        ("li_thickness",       0.02),
        ("cu_thickness",       0.3),
        ("water_thickness",    0.6),
        ("vacuum_thickness_1", 1.5),
        ("graphite_thickness", 0.7),
        ("vacuum_thickness_2", 0.48),
        ("ti_thickness",       0.6),
        ("air_gap",            0.1),
        ("wh_r",               50.0),
        ("eurofer_thickness",  0.5),
        ("tbm_width",          7.0),
        ("tbm_thickness",      2.0),
        ("tbm_height",         3.0),
        ("tbm_position_y",    -42.0),
    ])
    def test_target_param_as_uncertain_dict(self, param_name, default):
        """When a target/geometry param is declared uncertain, _get_mean extracts the mean."""
        uncertain_value = {"mean": default, "relative_stdev": 0.05, "pdf": "normal"}
        result = float(_get_mean(uncertain_value))
        assert result == pytest.approx(default)

    @pytest.mark.parametrize("param_name,default", [
        ("li_thickness",       0.02),
        ("cu_thickness",       0.3),
        ("water_thickness",    0.6),
        ("vacuum_thickness_1", 1.5),
        ("graphite_thickness", 0.7),
        ("vacuum_thickness_2", 0.48),
        ("ti_thickness",       0.6),
        ("air_gap",            0.1),
        ("wh_r",               50.0),
        ("eurofer_thickness",  0.5),
        ("tbm_width",          7.0),
        ("tbm_thickness",      2.0),
        ("tbm_height",         3.0),
        ("tbm_position_y",    -42.0),
    ])
    def test_target_param_as_scalar(self, param_name, default):
        """When a target/geometry param is a plain scalar, _get_mean returns it directly."""
        result = float(_get_mean(default))
        assert result == pytest.approx(default)

    def test_material_density_as_uncertain(self):
        """Material density declared uncertain should not crash."""
        val = {"mean": 3.43, "relative_stdev": 0.02, "pdf": "normal"}
        assert float(_get_mean(val)) == pytest.approx(3.43)

    def test_li6_enrichment_as_uncertain(self):
        """Li-6 enrichment declared uncertain should not crash."""
        val = {"mean": 7.5, "relative_stdev": 0.05, "pdf": "normal"}
        assert float(_get_mean(val)) == pytest.approx(7.5)
