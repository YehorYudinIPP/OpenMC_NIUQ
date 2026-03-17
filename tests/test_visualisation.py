"""
Tests for uq.visualisation – post-campaign plotting helpers.

Uses a lightweight mock of the EasyVVUQ analysis results object so that the
tests run without OpenMC or a real UQ campaign.
"""

import os
import shutil
import tempfile
import pickle

import numpy as np
import pytest

# ── Importable whether pytest is run from repo root or from tests/ ────────
import sys

_uq_dir = os.path.join(os.path.dirname(__file__), os.pardir, "uq")
if _uq_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_uq_dir))

from visualisation import (
    _QOI_UNITS,
    _unit_for_qoi,
    plot_input_uncertainty_pdfs,
    plot_qoi_distributions,
    plot_qoi_statistics,
    plot_qoi_statistics_table,
    plot_relative_std,
    plot_sobol_first_order_pie,
    plot_sobol_indices,
    plot_sobol_second_order_heatmap,
    visualise_results,
)
from visualise_uq_results import main as visualise_main


# ---------------------------------------------------------------------------
# Mock of EasyVVUQ analysis results
# ---------------------------------------------------------------------------

class _MockResults:
    """Minimal stand-in for an EasyVVUQ analysis results object."""

    def __init__(self, qois, param_names):
        self._qois = qois
        self._param_names = param_names
        self.raw_data = {q: np.random.normal(1.0, 0.1, 50) for q in qois}

    # -- describe() ---------------------------------------------------------
    def describe(self, qoi, stat=None):
        """Return a pandas-like Series or dict of descriptive statistics."""
        base = {
            "mean": np.array([1.0]),
            "std": np.array([0.1]),
            "10%": np.array([0.87]),
            "90%": np.array([1.13]),
        }
        if stat is not None:
            return base.get(stat, np.array([0.0]))
        return base

    # -- Sobol helpers ------------------------------------------------------
    def sobols_first(self, qoi=None):
        n = len(self._param_names)
        vals = np.random.dirichlet(np.ones(n))
        return {p: np.array([v]) for p, v in zip(self._param_names, vals)}

    def sobols_total(self, qoi=None):
        sobols = self.sobols_first(qoi)
        # Total ≥ first-order
        return {p: v + np.array([0.02]) for p, v in sobols.items()}

    def sobols_second(self, qoi=None):
        n = len(self._param_names)
        result = {}
        for i, pi in enumerate(self._param_names):
            row = {}
            for j, pj in enumerate(self._param_names):
                if i != j:
                    row[pj] = np.array([np.random.uniform(0.0, 0.05)])
            result[pi] = row
        return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QOIS = ["tritium_production_rate", "total_neutron_flux"]
PARAMS = [
    "li_ceramic_density",
    "li6_enrichment",
    "pebble_radius",
    "packing_fraction",
    "graphite_thickness",
]


@pytest.fixture()
def mock_results():
    return _MockResults(QOIS, PARAMS)


@pytest.fixture()
def distributions():
    """Dummy distributions dict (values unused – only keys matter)."""
    return {p: None for p in PARAMS}


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQoiUnits:
    """Verify that the QoI units dict and helper are available."""

    def test_all_qois_have_units(self):
        for qoi in QOIS:
            unit = _unit_for_qoi(qoi)
            assert isinstance(unit, str) and len(unit) > 0, (
                f"QoI '{qoi}' should have a non-empty unit string")

    def test_unknown_qoi_returns_empty(self):
        assert _unit_for_qoi("unknown_qoi") == ""

    def test_units_dict_has_all_known_qois(self):
        known = ["tritium_production_rate", "total_neutron_flux",
                 "tbm_incident_flux", "tbm_inner_flux",
                 "tbm_heating", "tbm_neutron_leakage"]
        for qoi in known:
            assert qoi in _QOI_UNITS


class TestPlotQoiStatistics:
    def test_creates_file(self, mock_results, tmp_dir):
        path = plot_qoi_statistics(mock_results, QOIS, tmp_dir)
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_file_nonempty(self, mock_results, tmp_dir):
        path = plot_qoi_statistics(mock_results, QOIS, tmp_dir)
        assert os.path.getsize(path) > 0

    def test_uses_describe_with_stat_arg(self, tmp_dir):
        """Ensure describe() is called with the stat argument (not dict-indexing)."""

        class _StrictDescribe(_MockResults):
            def describe(self, qoi, stat=None):
                if stat is None:
                    raise KeyError("describe() must be called with a stat argument")
                return super().describe(qoi, stat)

        res = _StrictDescribe(QOIS, PARAMS)
        path = plot_qoi_statistics(res, QOIS, tmp_dir)
        assert os.path.isfile(path)


class TestPlotSobolIndices:
    def test_creates_one_file_per_qoi(self, mock_results, distributions, tmp_dir):
        paths = plot_sobol_indices(mock_results, QOIS, distributions, tmp_dir)
        assert len(paths) == len(QOIS)
        for p in paths:
            assert os.path.isfile(p)

    def test_handles_missing_total(self, distributions, tmp_dir):
        """If sobols_total raises, only first-order bars should appear."""

        class _NoTotal(_MockResults):
            def sobols_total(self, qoi=None):
                raise RuntimeError("Not available for QMC")

        res = _NoTotal(QOIS, PARAMS)
        paths = plot_sobol_indices(res, QOIS, distributions, tmp_dir)
        assert len(paths) == len(QOIS)
        for p in paths:
            assert os.path.isfile(p)


class TestPlotQoiDistributions:
    def test_creates_file(self, mock_results, tmp_dir):
        path = plot_qoi_distributions(mock_results, QOIS, tmp_dir)
        assert path is not None
        assert os.path.isfile(path)

    def test_returns_none_when_no_raw_data(self, tmp_dir):
        res = _MockResults(QOIS, PARAMS)
        res.raw_data = None
        assert plot_qoi_distributions(res, QOIS, tmp_dir) is None


class TestPlotSobolSecondOrderHeatmap:
    def test_creates_one_file_per_qoi(self, mock_results, distributions, tmp_dir):
        paths = plot_sobol_second_order_heatmap(
            mock_results, QOIS, distributions, tmp_dir)
        assert len(paths) == len(QOIS)
        for p in paths:
            assert os.path.isfile(p)
            assert "sobol_second_order_" in p

    def test_returns_empty_when_unavailable(self, distributions, tmp_dir):
        class _NoSecond(_MockResults):
            def sobols_second(self, qoi=None):
                raise RuntimeError("Not available")

        res = _NoSecond(QOIS, PARAMS)
        paths = plot_sobol_second_order_heatmap(
            res, QOIS, distributions, tmp_dir)
        assert paths == []

    def test_diagonal_contains_first_order_indices(self, distributions, tmp_dir):
        """The main diagonal of the heatmap matrix must hold S1 values."""
        s1_called = []

        class _FixedSobols(_MockResults):
            def sobols_first(self, qoi=None):
                s1_called.append(qoi)
                # Deterministic first-order values
                return {p: np.array([0.1 * (i + 1)])
                        for i, p in enumerate(self._param_names)}

            def sobols_second(self, qoi=None):
                result = {}
                for pi in self._param_names:
                    row = {}
                    for pj in self._param_names:
                        if pi != pj:
                            row[pj] = np.array([0.001])
                    result[pi] = row
                return result

        res = _FixedSobols(QOIS, PARAMS)
        paths = plot_sobol_second_order_heatmap(
            res, QOIS, distributions, tmp_dir)
        assert len(paths) == len(QOIS)
        # sobols_first must have been called once per QoI
        assert len(s1_called) == len(QOIS)


class TestPlotSobolFirstOrderPie:
    def test_creates_one_file_per_qoi(self, mock_results, distributions, tmp_dir):
        paths = plot_sobol_first_order_pie(
            mock_results, QOIS, distributions, tmp_dir)
        assert len(paths) == len(QOIS)
        for p in paths:
            assert os.path.isfile(p)
            assert "sobol_pie_" in p


class TestPlotQoiStatisticsTable:
    def test_creates_file(self, mock_results, tmp_dir):
        path = plot_qoi_statistics_table(mock_results, QOIS, tmp_dir)
        assert os.path.isfile(path)
        assert "qoi_statistics_table" in path

    def test_file_nonempty(self, mock_results, tmp_dir):
        path = plot_qoi_statistics_table(mock_results, QOIS, tmp_dir)
        assert os.path.getsize(path) > 0


class TestPlotRelativeStd:
    def test_creates_file(self, mock_results, tmp_dir):
        path = plot_relative_std(mock_results, QOIS, tmp_dir)
        assert os.path.isfile(path)
        assert "qoi_relative_std" in path

    def test_file_nonempty(self, mock_results, tmp_dir):
        path = plot_relative_std(mock_results, QOIS, tmp_dir)
        assert os.path.getsize(path) > 0


class TestPlotInputUncertaintyPdfs:
    def test_returns_none_when_no_distributions(self, tmp_dir):
        dists = {p: None for p in PARAMS}
        result = plot_input_uncertainty_pdfs(dists, tmp_dir)
        assert result is None

    def test_creates_file_with_mock_distributions(self, tmp_dir):
        """Use a simple mock distribution that supports .sample()."""

        class _MockDist:
            def sample(self, n):
                return np.random.normal(1.0, 0.1, n)

        dists = {PARAMS[0]: _MockDist(), PARAMS[1]: _MockDist()}
        path = plot_input_uncertainty_pdfs(dists, tmp_dir)
        assert path is not None
        assert os.path.isfile(path)
        assert "input_uncertainty_pdfs" in path


class TestVisualiseResults:
    def test_creates_output_dir(self, mock_results, distributions, tmp_dir):
        out = os.path.join(tmp_dir, "my_plots")
        files = visualise_results(mock_results, QOIS, distributions,
                                  output_dir=out)
        assert os.path.isdir(out)
        assert len(files) > 0

    def test_default_dir_name(self, mock_results, distributions):
        """When output_dir is None, a timestamped folder is created."""
        files = visualise_results(mock_results, QOIS, distributions,
                                  timestamp="TEST_TS")
        out = "plots_openmc_uq_TEST_TS"
        try:
            assert os.path.isdir(out)
            assert len(files) > 0
        finally:
            shutil.rmtree(out, ignore_errors=True)


class TestVisualiseUqResultsScript:
    """Tests for the standalone visualise_uq_results.py script."""

    def test_main_with_results_pickle(self, mock_results, tmp_dir):
        """main() loads a pickled results file and generates plots."""
        pkl_path = os.path.join(tmp_dir, "results.pickle")
        with open(pkl_path, "wb") as fh:
            pickle.dump(mock_results, fh)

        out_dir = os.path.join(tmp_dir, "standalone_plots")
        files = visualise_main(pkl_path, output_dir=out_dir)
        assert os.path.isdir(out_dir)
        assert len(files) > 0
        for f in files:
            assert os.path.isfile(f)
            assert f.endswith(".png")

    def test_main_with_config_pickle(self, mock_results, tmp_dir):
        """main() accepts an optional campaign config pickle."""
        results_pkl = os.path.join(tmp_dir, "results.pickle")
        with open(results_pkl, "wb") as fh:
            pickle.dump(mock_results, fh)

        # Config dict that matches model_config.yaml structure
        config = {
            "geometry": {
                "pebble_radius": {"mean": 0.1, "relative_stdev": 0.05, "pdf": "normal"},
                "packing_fraction": {"mean": 0.3, "relative_stdev": 0.05, "pdf": "uniform"},
                "graphite_thickness": {"mean": 0.7, "relative_stdev": 0.05, "pdf": "normal"},
            },
            "materials": {
                "li_ceramic": {
                    "density": {"mean": 3.43, "relative_stdev": 0.02, "pdf": "normal"},
                    "li6_enrichment": {"mean": 7.5, "relative_stdev": 0.05, "pdf": "normal"},
                },
            },
        }
        config_pkl = os.path.join(tmp_dir, "config.pickle")
        with open(config_pkl, "wb") as fh:
            pickle.dump(config, fh)

        out_dir = os.path.join(tmp_dir, "standalone_plots_cfg")
        files = visualise_main(results_pkl, config_file=config_pkl,
                               output_dir=out_dir)
        assert os.path.isdir(out_dir)
        assert len(files) > 0
