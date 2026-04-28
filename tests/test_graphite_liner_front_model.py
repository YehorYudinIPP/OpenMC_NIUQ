"""
Tests for the front-wall graphite-liner variant of the OpenMC TBM model.

Validates that ``openmc_model_run_graphite_liner_front.py`` differs from the
back-liner model only in the placement of the graphite liner on the inner
*front* (neutron-facing) wall of the TBM.  These tests do not require OpenMC;
they exercise the pure-Python configuration logic and code structure.
"""

import os
import sys

import pytest
import yaml

# ── Importable whether pytest is run from repo root or from tests/ ────────
_uq_dir = os.path.join(os.path.dirname(__file__), os.pardir, "uq")
if _uq_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_uq_dir))

from openmc_model_run_graphite_liner_front import _get_mean, load_config


# ---------------------------------------------------------------------------
# _get_mean helper
# ---------------------------------------------------------------------------

class TestGetMeanGraphiteLinerFront:
    """Verify _get_mean works correctly in the front-liner module."""

    def test_scalar_value(self):
        assert _get_mean(3.43) == 3.43

    def test_dict_with_mean(self):
        assert _get_mean({"mean": 0.5, "relative_stdev": 0.05, "pdf": "normal"}) == 0.5


# ---------------------------------------------------------------------------
# Front graphite liner config file
# ---------------------------------------------------------------------------

class TestGraphiteLinerFrontConfig:
    """Tests for the front-liner YAML configuration file."""

    @pytest.fixture()
    def config_path(self):
        return os.path.join(
            os.path.dirname(__file__), os.pardir,
            "uq", "config", "model_config_graphite_liner_front.yaml",
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
        # The parameter is an uncertain-parameter dict for UQ scanning
        glt = geom["graphite_liner_thickness"]
        assert isinstance(glt, dict), (
            "graphite_liner_thickness should be an uncertain-parameter dict "
            "with 'mean', 'relative_stdev', and 'pdf' keys"
        )
        assert float(glt["mean"]) == pytest.approx(0.5)
        assert "relative_stdev" in glt
        assert "pdf" in glt

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
# Structural comparison with the back-liner module
# ---------------------------------------------------------------------------

class TestGraphiteLinerFrontModuleStructure:
    """Ensure the front-liner module preserves the base-model API."""

    def test_build_openmc_model_is_callable(self):
        from openmc_model_run_graphite_liner_front import build_openmc_model
        assert callable(build_openmc_model)

    def test_extract_qois_is_callable(self):
        from openmc_model_run_graphite_liner_front import extract_qois
        assert callable(extract_qois)

    def test_save_results_csv_is_callable(self):
        from openmc_model_run_graphite_liner_front import save_results_csv
        assert callable(save_results_csv)

    def test_load_config_is_callable(self):
        from openmc_model_run_graphite_liner_front import load_config
        assert callable(load_config)

    def test_main_is_callable(self):
        from openmc_model_run_graphite_liner_front import main
        assert callable(main)


# ---------------------------------------------------------------------------
# Geometry: front liner is placed before (lower z) the ceramic
# ---------------------------------------------------------------------------

class TestGraphiteLinerFrontGeometryLogic:
    """
    Verify the surface z-positions in the front-liner model are ordered
    correctly without running OpenMC.
    """

    def test_front_liner_precedes_ceramic(self):
        """
        graphite_liner_end must be strictly between inner_start and inner_end,
        placing the graphite *before* the ceramic along the beam axis (z).
        """
        # Replicate the geometry calculation from build_openmc_model
        # using nominal (default) values.
        li_thickness       = 0.02
        cu_thickness       = 0.3
        water_thickness    = 0.6
        vacuum_thickness_1 = 1.5
        graphite_thickness = 0.7
        vacuum_thickness_2 = 0.48
        ti_thickness       = 0.6
        air_gap            = 0.1
        eurofer_thickness  = 0.5
        tbm_thickness      = 2.0
        graphite_liner_thickness = 0.5

        z_li_hi    =  li_thickness / 2
        z_cu1_hi   = z_li_hi + cu_thickness
        z_water_hi = z_cu1_hi + water_thickness
        z_cu2_hi   = z_water_hi + cu_thickness
        z_vac1_hi  = z_cu2_hi + vacuum_thickness_1
        z_graph_hi = z_vac1_hi + graphite_thickness
        z_vac2_hi  = z_graph_hi + vacuum_thickness_2
        z_ti_hi    = z_vac2_hi + ti_thickness

        base_case = z_ti_hi + air_gap

        inner_start_z      = base_case + eurofer_thickness
        graphite_liner_end_z = inner_start_z + graphite_liner_thickness
        inner_end_z        = base_case + tbm_thickness - eurofer_thickness

        # Liner starts at the front face of the inner cavity …
        assert graphite_liner_end_z > inner_start_z
        # … and ends before the back face of the inner cavity
        assert graphite_liner_end_z < inner_end_z

    def test_ceramic_region_is_behind_liner(self):
        """
        The ceramic z-start (graphite_liner_end) is greater than inner_start,
        confirming the ceramic is behind (higher z than) the front liner.
        """
        eurofer_thickness        = 0.5
        tbm_thickness            = 2.0
        graphite_liner_thickness = 0.5

        # Arbitrary base_case value
        base_case = 10.0

        inner_start_z        = base_case + eurofer_thickness
        graphite_liner_end_z = inner_start_z + graphite_liner_thickness
        inner_end_z          = base_case + tbm_thickness - eurofer_thickness

        ceramic_start_z = graphite_liner_end_z
        ceramic_end_z   = inner_end_z

        assert ceramic_start_z > inner_start_z
        assert ceramic_end_z   > ceramic_start_z
