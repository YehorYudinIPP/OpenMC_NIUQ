"""
Tests for uq.visualisation – post-campaign plotting helpers.

Uses a lightweight mock of the EasyVVUQ analysis results object so that the
tests run without OpenMC or a real UQ campaign.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

# ── Importable whether pytest is run from repo root or from tests/ ────────
import sys

_uq_dir = os.path.join(os.path.dirname(__file__), os.pardir, "uq")
if _uq_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_uq_dir))

from visualisation import (
    plot_qoi_distributions,
    plot_qoi_statistics,
    plot_sobol_indices,
    visualise_results,
)


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

class TestPlotQoiStatistics:
    def test_creates_file(self, mock_results, tmp_dir):
        path = plot_qoi_statistics(mock_results, QOIS, tmp_dir)
        assert os.path.isfile(path)
        assert path.endswith(".png")

    def test_file_nonempty(self, mock_results, tmp_dir):
        path = plot_qoi_statistics(mock_results, QOIS, tmp_dir)
        assert os.path.getsize(path) > 0


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
