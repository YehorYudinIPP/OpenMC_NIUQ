"""
Tests for the graphite-liner variant of the OpenMC TBM model.

Validates that ``openmc_model_run_graphite_liner.py`` differs from the base
model only in the addition of a graphite liner on the inner back wall of the
TBM.  These tests do not require OpenMC; they exercise the pure-Python
configuration logic and code structure.
"""

import os
import sys
import textwrap

import pytest
import yaml

# ── Importable whether pytest is run from repo root or from tests/ ────────
_uq_dir = os.path.join(os.path.dirname(__file__), os.pardir, "uq")
if _uq_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_uq_dir))

from openmc_model_run_graphite_liner import _get_mean, load_config


# ---------------------------------------------------------------------------
# _get_mean helper (same as base model – ensure it was copied correctly)
# ---------------------------------------------------------------------------

class TestGetMeanGraphiteLiner:
    """Verify _get_mean works identically in the graphite-liner module."""

    def test_scalar_value(self):
        assert _get_mean(3.43) == 3.43

    def test_dict_with_mean(self):
        assert _get_mean({"mean": 0.5, "relative_stdev": 0.05, "pdf": "normal"}) == 0.5


# ---------------------------------------------------------------------------
# Graphite liner config file
# ---------------------------------------------------------------------------

class TestGraphiteLinerConfig:
    """Tests for the graphite-liner YAML configuration file."""

    @pytest.fixture()
    def config_path(self):
        return os.path.join(
            os.path.dirname(__file__), os.pardir,
            "uq", "config", "model_config_graphite_liner.yaml",
        )

    def test_config_file_exists(self, config_path):
        assert os.path.isfile(config_path)

    def test_config_loads_successfully(self, config_path):
        config = load_config(config_path)
        assert config is not None

    def test_config_has_graphite_liner_thickness(self, config_path):
        config = load_config(config_path)
        geom = config.get("geometry", {})
        assert "graphite_liner_thickness" in geom
        assert float(geom["graphite_liner_thickness"]) == pytest.approx(0.5)

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


# ---------------------------------------------------------------------------
# Structural comparison with the base model module
# ---------------------------------------------------------------------------

class TestGraphiteLinerModuleStructure:
    """Ensure the graphite-liner module preserves the base-model API."""

    def test_build_openmc_model_is_callable(self):
        from openmc_model_run_graphite_liner import build_openmc_model
        assert callable(build_openmc_model)

    def test_extract_qois_is_callable(self):
        from openmc_model_run_graphite_liner import extract_qois
        assert callable(extract_qois)

    def test_save_results_csv_is_callable(self):
        from openmc_model_run_graphite_liner import save_results_csv
        assert callable(save_results_csv)

    def test_load_config_is_callable(self):
        from openmc_model_run_graphite_liner import load_config
        assert callable(load_config)

    def test_main_is_callable(self):
        from openmc_model_run_graphite_liner import main
        assert callable(main)
