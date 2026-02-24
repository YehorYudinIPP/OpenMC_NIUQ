"""
Tests for config parameter reading, auto-discovery of uncertain parameters,
logging setup, and the _get_mean() helper.

These tests do not require OpenMC or EasyVVUQ; they exercise the pure-Python
configuration logic.
"""

import logging
import os
import sys
import tempfile
import warnings

import pytest
import yaml

# ── Importable whether pytest is run from repo root or from tests/ ────────
_uq_dir = os.path.join(os.path.dirname(__file__), os.pardir, "uq")
if _uq_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_uq_dir))

from openmc_model_run import _get_mean


# ---------------------------------------------------------------------------
# _get_mean helper
# ---------------------------------------------------------------------------

class TestGetMean:
    """Tests for the _get_mean() helper in openmc_model_run.py."""

    def test_scalar_value(self):
        assert _get_mean(3.43) == 3.43

    def test_dict_with_mean(self):
        assert _get_mean({"mean": 7.5, "relative_stdev": 0.05, "pdf": "normal"}) == 7.5

    def test_dict_without_mean(self):
        """A dict without 'mean' should return 0.0 as fallback."""
        assert _get_mean({"relative_stdev": 0.05}) == 0.0

    def test_int_scalar(self):
        assert _get_mean(10) == 10

    def test_zero(self):
        assert _get_mean(0.0) == 0.0


# ---------------------------------------------------------------------------
# Auto-discovery of uncertain parameters
# ---------------------------------------------------------------------------

class TestDiscoverUncertainParameters:
    """Tests for discover_uncertain_parameters() in easyvvuq_openmc.py."""

    @pytest.fixture()
    def sample_config(self):
        return {
            "geometry": {
                "type": "tbm_pebble_bed",
                "wh_r": 50.0,  # fixed – plain scalar
                "graphite_thickness": {
                    "mean": 0.7,
                    "relative_stdev": 0.05,
                    "pdf": "normal",
                },
                "pebble_radius": {
                    "mean": 0.10,
                    "relative_stdev": 0.05,
                    "pdf": "normal",
                },
                "packing_fraction": {
                    "mean": 0.30,
                    "relative_stdev": 0.05,
                    "pdf": "uniform",
                },
            },
            "materials": {
                "li_ceramic": {
                    "name": "Li2TiO3",
                    "density": {
                        "mean": 3.43,
                        "relative_stdev": 0.02,
                        "pdf": "normal",
                    },
                    "li6_enrichment": {
                        "mean": 7.5,
                        "relative_stdev": 0.05,
                        "pdf": "normal",
                    },
                },
            },
            "output": {
                "qoi": ["tritium_production_rate", "total_neutron_flux"],
            },
        }

    def test_discovers_all_uncertain_params(self, sample_config):
        from easyvvuq_openmc import discover_uncertain_parameters
        found = discover_uncertain_parameters(sample_config)
        names = {p["name"] for p in found}
        assert "geometry_graphite_thickness" in names or "graphite_thickness" in names
        assert len(found) == 5  # 3 geometry + 2 material

    def test_fixed_scalars_are_excluded(self, sample_config):
        from easyvvuq_openmc import discover_uncertain_parameters
        found = discover_uncertain_parameters(sample_config)
        names = {p["name"] for p in found}
        assert "wh_r" not in names
        assert "type" not in names

    def test_each_spec_has_required_keys(self, sample_config):
        from easyvvuq_openmc import discover_uncertain_parameters
        found = discover_uncertain_parameters(sample_config)
        for p in found:
            assert "name" in p
            assert "path" in p
            assert "mean" in p
            assert "relative_stdev" in p
            assert "pdf" in p


# ---------------------------------------------------------------------------
# Warning for missing pdf key
# ---------------------------------------------------------------------------

class TestMissingPdfWarning:
    """When 'pdf' is missing from a distribution spec, a warning should fire."""

    def test_warns_on_missing_pdf(self):
        config = {
            "geometry": {
                "some_param": {
                    "mean": 1.0,
                    "relative_stdev": 0.03,
                    # no 'pdf' key
                },
            },
        }
        from easyvvuq_openmc import discover_uncertain_parameters
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            specs = discover_uncertain_parameters(config)
            assert len(w) == 1
            assert "no 'pdf'" in str(w[0].message).lower()
        # Default should be 'uniform'
        assert specs[0]["pdf"] == "uniform"


# ---------------------------------------------------------------------------
# define_model_parameters reads QoIs from config
# ---------------------------------------------------------------------------

class TestDefineModelParameters:
    def test_reads_qois_from_config(self):
        config = {
            "geometry": {
                "pebble_radius": {"mean": 0.1, "relative_stdev": 0.05, "pdf": "normal"},
            },
            "output": {
                "qoi": ["tritium_production_rate", "tbm_heating"],
            },
        }
        from easyvvuq_openmc import define_model_parameters
        params, qois, specs = define_model_parameters(config)
        assert "tritium_production_rate" in qois
        assert "tbm_heating" in qois
        assert len(qois) == 2

    def test_default_qois_when_not_in_config(self):
        config = {
            "geometry": {
                "pebble_radius": {"mean": 0.1, "relative_stdev": 0.05, "pdf": "normal"},
            },
        }
        from easyvvuq_openmc import define_model_parameters
        _, qois, _ = define_model_parameters(config)
        assert "tritium_production_rate" in qois
        assert "total_neutron_flux" in qois


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

class TestLogging:
    def test_setup_logging_creates_file(self):
        import tempfile
        from easyvvuq_openmc import setup_logging
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            returned = setup_logging(log_file=log_path)
            assert returned == log_path
            logger = logging.getLogger("openmc_uq")
            logger.info("test message")
            # Flush handlers
            for h in logging.getLogger().handlers:
                h.flush()
            with open(log_path) as fh:
                content = fh.read()
            assert "test message" in content
        finally:
            os.unlink(log_path)
            # Clean up handlers
            for h in logging.getLogger().handlers[:]:
                logging.getLogger().removeHandler(h)


# ---------------------------------------------------------------------------
# Config reading: scalar treated as uncertain
# ---------------------------------------------------------------------------

class TestScalarAsUncertain:
    """When a scalar is the only value but is discovered, it should work."""

    def test_empty_config_no_uncertain(self):
        config = {"geometry": {"wh_r": 50.0}}
        from easyvvuq_openmc import discover_uncertain_parameters
        found = discover_uncertain_parameters(config)
        assert len(found) == 0


# ---------------------------------------------------------------------------
# define_parameter_distributions
# ---------------------------------------------------------------------------

class TestDefineParameterDistributions:
    def test_builds_distributions(self):
        config = {
            "geometry": {
                "pebble_radius": {"mean": 0.1, "relative_stdev": 0.05, "pdf": "normal"},
                "packing_fraction": {"mean": 0.3, "relative_stdev": 0.05, "pdf": "uniform"},
            },
        }
        from easyvvuq_openmc import define_parameter_distributions
        dists = define_parameter_distributions(config)
        assert len(dists) == 2

    def test_cov_override(self):
        config = {
            "geometry": {
                "pebble_radius": {"mean": 0.1, "relative_stdev": 0.05, "pdf": "normal"},
            },
        }
        from easyvvuq_openmc import define_parameter_distributions
        dists = define_parameter_distributions(config, cov_override=0.10)
        assert len(dists) == 1

    def test_unsupported_distribution_raises(self):
        config = {
            "geometry": {
                "pebble_radius": {"mean": 0.1, "relative_stdev": 0.05, "pdf": "beta"},
            },
        }
        from easyvvuq_openmc import define_parameter_distributions
        with pytest.raises(ValueError, match="Unsupported distribution"):
            define_parameter_distributions(config)
