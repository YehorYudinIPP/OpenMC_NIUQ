"""
Tests for the parametric model builder and parameter_overrides utilities.

These tests do not require OpenMC or EasyVVUQ — they exercise the pure-Python
parameter-override, config-merging, sampled-value substitution, and shared-XML
identification logic.
"""

import os
import sys
import tempfile
import warnings

import pytest
import yaml

# ── Make the uq package importable regardless of working directory ────────
_repo_root = os.path.join(os.path.dirname(__file__), os.pardir)
if os.path.abspath(_repo_root) not in sys.path:
    sys.path.insert(0, os.path.abspath(_repo_root))

from uq.util.parameter_overrides import (
    _get_nested,
    _set_nested,
    apply_sampled_values,
    identify_shared_xml,
    load_uq_parameters,
    merge_parameters,
    resolve_base_config_path,
)
from uq.parametric_model_builder import ParametricModelBuilder


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def base_config():
    """Minimal base configuration dict."""
    return {
        "geometry": {
            "type": "tbm_pebble_bed",
            "wh_r": 50.0,
            "graphite_thickness": 0.7,
            "pebble_radius": 0.10,
        },
        "materials": {
            "li_ceramic": {
                "name": "Li2TiO3",
                "density": 3.43,
                "li6_enrichment": 7.5,
            },
        },
        "simulation": {
            "particles": 10000,
            "batches": 100,
        },
        "output": {
            "qoi": ["tritium_production_rate"],
        },
    }


@pytest.fixture()
def uq_parameters():
    """Minimal uncertain-parameter spec dict."""
    return {
        "graphite_thickness": {
            "path": "geometry.graphite_thickness",
            "mean": 0.7,
            "relative_stdev": 0.05,
            "pdf": "normal",
        },
        "li_ceramic_density": {
            "path": "materials.li_ceramic.density",
            "mean": 3.43,
            "relative_stdev": 0.02,
            "pdf": "normal",
        },
    }


@pytest.fixture()
def base_config_file(base_config, tmp_path):
    """Write the base config to a temp YAML file and return its path."""
    path = tmp_path / "model_config.yaml"
    with open(path, "w") as fh:
        yaml.dump(base_config, fh)
    return str(path)


@pytest.fixture()
def uq_yaml_file(uq_parameters, tmp_path):
    """Write a parameters-only YAML and matching base config into tmp_path."""
    # Write the base config first
    base_cfg = {
        "geometry": {"wh_r": 50.0, "graphite_thickness": 0.7},
        "materials": {"li_ceramic": {"density": 3.43}},
        "simulation": {"particles": 10000},
        "output": {"qoi": ["tritium_production_rate"]},
    }
    base_path = tmp_path / "model_config.yaml"
    with open(base_path, "w") as fh:
        yaml.dump(base_cfg, fh)

    # Write the UQ parameters YAML referencing the base config
    uq_data = {
        "base_config": "model_config.yaml",
        "parameters": uq_parameters,
    }
    uq_path = tmp_path / "uq_parameters.yaml"
    with open(uq_path, "w") as fh:
        yaml.dump(uq_data, fh)

    return str(uq_path)


# ═══════════════════════════════════════════════════════════════════════════
# Tests for _set_nested / _get_nested
# ═══════════════════════════════════════════════════════════════════════════

class TestNestedHelpers:
    def test_set_nested_simple(self):
        d = {"a": {"b": 1}}
        _set_nested(d, "a.b", 42)
        assert d["a"]["b"] == 42

    def test_set_nested_creates_intermediate(self):
        d = {}
        _set_nested(d, "x.y.z", "hello")
        assert d["x"]["y"]["z"] == "hello"

    def test_get_nested_existing(self):
        d = {"a": {"b": {"c": 99}}}
        assert _get_nested(d, "a.b.c") == 99

    def test_get_nested_missing(self):
        d = {"a": {"b": 1}}
        assert _get_nested(d, "a.x.y", default="nope") == "nope"

    def test_get_nested_default(self):
        assert _get_nested({}, "missing", default=42) == 42


# ═══════════════════════════════════════════════════════════════════════════
# Tests for load_uq_parameters
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadUqParameters:
    def test_loads_valid_file(self, uq_yaml_file):
        data = load_uq_parameters(uq_yaml_file)
        assert "parameters" in data
        assert "graphite_thickness" in data["parameters"]

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_uq_parameters("/nonexistent/path.yaml")

    def test_raises_on_missing_parameters_key(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        with open(bad_file, "w") as fh:
            yaml.dump({"base_config": "something.yaml"}, fh)
        with pytest.raises(ValueError, match="'parameters'"):
            load_uq_parameters(str(bad_file))


# ═══════════════════════════════════════════════════════════════════════════
# Tests for resolve_base_config_path
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveBaseConfigPath:
    def test_relative_path(self, tmp_path):
        uq_path = str(tmp_path / "subdir" / "params.yaml")
        result = resolve_base_config_path(uq_path, "model_config.yaml")
        expected = os.path.normpath(
            os.path.join(str(tmp_path), "subdir", "model_config.yaml")
        )
        assert result == expected

    def test_absolute_path(self):
        result = resolve_base_config_path("/any/path.yaml", "/abs/config.yaml")
        assert result == "/abs/config.yaml"


# ═══════════════════════════════════════════════════════════════════════════
# Tests for merge_parameters
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeParameters:
    def test_merges_uq_spec_into_base(self, base_config, uq_parameters):
        merged = merge_parameters(base_config, uq_parameters)

        # graphite_thickness should now be a UQ-spec dict
        gt = merged["geometry"]["graphite_thickness"]
        assert isinstance(gt, dict)
        assert gt["mean"] == 0.7
        assert gt["pdf"] == "normal"

    def test_does_not_modify_base(self, base_config, uq_parameters):
        original_gt = base_config["geometry"]["graphite_thickness"]
        merge_parameters(base_config, uq_parameters)
        # base_config must be untouched
        assert base_config["geometry"]["graphite_thickness"] == original_gt

    def test_preserves_fixed_params(self, base_config, uq_parameters):
        merged = merge_parameters(base_config, uq_parameters)
        assert merged["geometry"]["wh_r"] == 50.0
        assert merged["simulation"]["particles"] == 10000

    def test_warns_on_missing_path(self, base_config):
        bad_params = {"bad": {"mean": 1.0}}  # no 'path'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            merge_parameters(base_config, bad_params)
            assert len(w) == 1
            assert "missing" in str(w[0].message).lower()

    def test_default_relative_stdev_and_pdf(self, base_config):
        params = {
            "pebble_radius": {
                "path": "geometry.pebble_radius",
                "mean": 0.1,
            },
        }
        merged = merge_parameters(base_config, params)
        pr = merged["geometry"]["pebble_radius"]
        assert pr["relative_stdev"] == 0.05
        assert pr["pdf"] == "uniform"


# ═══════════════════════════════════════════════════════════════════════════
# Tests for apply_sampled_values
# ═══════════════════════════════════════════════════════════════════════════

class TestApplySampledValues:
    def test_substitutes_mean(self, base_config, uq_parameters):
        merged = merge_parameters(base_config, uq_parameters)
        sampled = {"graphite_thickness": 0.72}
        result = apply_sampled_values(merged, sampled, uq_parameters)
        assert result["geometry"]["graphite_thickness"]["mean"] == 0.72

    def test_substitutes_scalar(self, base_config):
        """When the target is a plain scalar (not a dict with 'mean')."""
        params = {
            "wh_r": {"path": "geometry.wh_r", "mean": 50.0},
        }
        sampled = {"wh_r": 55.0}
        result = apply_sampled_values(base_config, sampled, params)
        assert result["geometry"]["wh_r"] == 55.0

    def test_does_not_modify_original(self, base_config, uq_parameters):
        merged = merge_parameters(base_config, uq_parameters)
        sampled = {"graphite_thickness": 0.72}
        apply_sampled_values(merged, sampled, uq_parameters)
        # merged should still have the original mean
        assert merged["geometry"]["graphite_thickness"]["mean"] == 0.7

    def test_warns_on_unknown_param(self, base_config, uq_parameters):
        merged = merge_parameters(base_config, uq_parameters)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_sampled_values(merged, {"unknown_param": 1.0}, uq_parameters)
            assert len(w) == 1
            assert "not found" in str(w[0].message).lower()


# ═══════════════════════════════════════════════════════════════════════════
# Tests for identify_shared_xml
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentifySharedXml:
    def test_geometry_only(self):
        params = {
            "graphite_thickness": {"path": "geometry.graphite_thickness"},
        }
        shared, varying = identify_shared_xml(params)
        assert "geometry.xml" in varying
        assert "settings.xml" in shared
        assert "tallies.xml" in shared
        assert "materials.xml" in shared

    def test_mixed_sections(self, uq_parameters):
        shared, varying = identify_shared_xml(uq_parameters)
        assert "geometry.xml" in varying
        assert "materials.xml" in varying
        assert "settings.xml" in shared
        assert "tallies.xml" in shared

    def test_no_params(self):
        shared, varying = identify_shared_xml({})
        assert len(varying) == 0
        assert shared == {"geometry.xml", "materials.xml",
                          "settings.xml", "tallies.xml"}


# ═══════════════════════════════════════════════════════════════════════════
# Tests for ParametricModelBuilder
# ═══════════════════════════════════════════════════════════════════════════

class TestParametricModelBuilder:
    def test_from_yaml(self, uq_yaml_file):
        builder = ParametricModelBuilder.from_yaml(uq_yaml_file)
        assert "graphite_thickness" in builder.parameter_names
        assert "li_ceramic_density" in builder.parameter_names

    def test_from_python(self, base_config_file, uq_parameters):
        builder = ParametricModelBuilder(
            base_config_path=base_config_file,
            parameters=uq_parameters,
        )
        assert len(builder.parameter_names) == 2

    def test_build_config_no_samples(self, base_config_file, uq_parameters):
        builder = ParametricModelBuilder(base_config_file, uq_parameters)
        config = builder.build_config()
        gt = config["geometry"]["graphite_thickness"]
        assert isinstance(gt, dict)
        assert gt["mean"] == 0.7

    def test_build_config_with_samples(self, base_config_file, uq_parameters):
        builder = ParametricModelBuilder(base_config_file, uq_parameters)
        config = builder.build_config(
            sampled_values={"graphite_thickness": 0.75}
        )
        assert config["geometry"]["graphite_thickness"]["mean"] == 0.75

    def test_shared_xml_files(self, base_config_file, uq_parameters):
        builder = ParametricModelBuilder(base_config_file, uq_parameters)
        shared, varying = builder.shared_xml_files()
        assert "settings.xml" in shared
        assert "tallies.xml" in shared
        assert "geometry.xml" in varying

    def test_cache_and_restore_shared_xml(self, base_config_file,
                                          uq_parameters, tmp_path):
        builder = ParametricModelBuilder(base_config_file, uq_parameters)

        # Simulate exported XML files
        source = tmp_path / "source"
        source.mkdir()
        for name in ["geometry.xml", "materials.xml", "settings.xml",
                      "tallies.xml"]:
            (source / name).write_text(f"<{name}/>")

        # Cache shared files
        cache = tmp_path / "cache"
        cached = builder.cache_shared_xml(str(source), str(cache))
        assert len(cached) >= 1
        assert all(os.path.exists(p) for p in cached)

        # Restore to a new run directory
        run_dir = tmp_path / "run_001"
        restored = builder.restore_shared_xml(str(cache), str(run_dir))
        assert len(restored) == len(cached)
        assert all(os.path.exists(p) for p in restored)

    def test_raises_on_missing_base_config(self, uq_parameters):
        with pytest.raises(FileNotFoundError):
            ParametricModelBuilder("/nonexistent.yaml", uq_parameters)

    def test_repr(self, base_config_file, uq_parameters):
        builder = ParametricModelBuilder(base_config_file, uq_parameters)
        r = repr(builder)
        assert "ParametricModelBuilder" in r
        assert "model_config.yaml" in r

    def test_from_yaml_missing_base_config_key(self, tmp_path):
        uq_path = tmp_path / "bad_uq.yaml"
        with open(uq_path, "w") as fh:
            yaml.dump({"parameters": {"x": {"path": "a.b", "mean": 1}}}, fh)
        with pytest.raises(ValueError, match="base_config"):
            ParametricModelBuilder.from_yaml(str(uq_path))

    def test_properties(self, base_config_file, uq_parameters):
        builder = ParametricModelBuilder(base_config_file, uq_parameters)
        assert os.path.isabs(builder.base_config_path)
        params = builder.parameters
        assert isinstance(params, dict)
        # Mutating the returned dict should not affect the builder
        params["extra"] = {}
        assert "extra" not in builder.parameters
