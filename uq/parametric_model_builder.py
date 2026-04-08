"""
parametric_model_builder.py – Construct UQ models from parameter-only specs.

Provides a :class:`ParametricModelBuilder` that separates the UQ parameter
specification from the full model configuration.  Users specify **only** the
uncertain parameters (in a YAML file or a Python dict); the rest of the
model comes from a base configuration file.

Key features
------------
* **Parameters-only specification** – no need to duplicate the full config.
* **Sampled-value substitution** – replace parameter means with sampled
  values during the per-run construction phase.
* **Shared-XML caching** – XML files that do not depend on any uncertain
  parameter are generated once and reused across all UQ runs.

Usage
-----
From a YAML file::

    builder = ParametricModelBuilder.from_yaml("uq/config/uq_parameters.yaml")
    config  = builder.build_config(sampled_values={"graphite_thickness": 0.72})
    shared, varying = builder.shared_xml_files()

From a Python script::

    builder = ParametricModelBuilder(
        base_config_path="uq/config/model_config.yaml",
        parameters={
            "graphite_thickness": {
                "path": "geometry.graphite_thickness",
                "mean": 0.7,
                "relative_stdev": 0.05,
                "pdf": "normal",
            },
        },
    )
    config  = builder.build_config()
"""

import os
import shutil

import yaml

from uq.util.parameter_overrides import (
    apply_sampled_values,
    identify_shared_xml,
    load_uq_parameters,
    merge_parameters,
    resolve_base_config_path,
)
from uq.util.utils import load_config


class ParametricModelBuilder:
    """
    Build OpenMC model configurations by overlaying a small set of
    uncertain parameters onto a base (full) model configuration.

    Parameters
    ----------
    base_config_path : str
        Path to the full model YAML configuration file.
    parameters : dict
        Mapping of parameter short names to their specs.  Each spec must
        contain at least ``path`` (dot-separated YAML path) and ``mean``.
        Optional: ``relative_stdev`` (default 0.05), ``pdf`` (default
        ``"uniform"``).
    """

    def __init__(self, base_config_path, parameters):
        if not os.path.exists(base_config_path):
            raise FileNotFoundError(
                f"Base config not found: {base_config_path}"
            )
        self._base_config_path = os.path.abspath(base_config_path)
        self._parameters = dict(parameters)
        self._base_config = load_config(self._base_config_path)
        if self._base_config is None:
            raise ValueError(
                f"Failed to load base configuration from: {base_config_path}"
            )

    # -----------------------------------------------------------------
    # Alternate constructors
    # -----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, uq_yaml_path):
        """
        Create a builder from a parameters-only YAML file.

        The YAML file must contain ``base_config`` (path to the full model
        config, resolved relative to the YAML file) and ``parameters``.

        Parameters
        ----------
        uq_yaml_path : str
            Path to the parameters-only YAML file.

        Returns
        -------
        ParametricModelBuilder
        """
        data = load_uq_parameters(uq_yaml_path)

        base_rel = data.get("base_config")
        if base_rel is None:
            raise ValueError(
                f"UQ YAML '{uq_yaml_path}' must contain a 'base_config' key."
            )

        base_path = resolve_base_config_path(uq_yaml_path, base_rel)
        parameters = data["parameters"]

        return cls(base_config_path=base_path, parameters=parameters)

    # -----------------------------------------------------------------
    # Configuration building
    # -----------------------------------------------------------------

    def build_config(self, sampled_values=None):
        """
        Return a complete model configuration with uncertain parameters
        merged in and, optionally, sampled values substituted.

        Parameters
        ----------
        sampled_values : dict or None
            When provided, each key should match a parameter short name and
            the corresponding value is the sampled float.  The ``mean``
            field of the matching parameter node is replaced with this value.

        Returns
        -------
        dict
            A new configuration dict ready for model construction.
        """
        merged = merge_parameters(self._base_config, self._parameters)

        if sampled_values:
            merged = apply_sampled_values(
                merged, sampled_values, self._parameters
            )

        return merged

    # -----------------------------------------------------------------
    # Shared-XML helpers
    # -----------------------------------------------------------------

    def shared_xml_files(self):
        """
        Identify which OpenMC XML files are shared (invariant) across runs.

        Returns
        -------
        tuple[set, set]
            ``(shared, varying)`` – sets of XML file names.
        """
        return identify_shared_xml(self._parameters)

    def cache_shared_xml(self, source_dir, cache_dir):
        """
        Copy shared (invariant) XML files from *source_dir* to *cache_dir*.

        Call this **once** after the first model export to avoid
        regenerating files that do not change between UQ runs.

        Parameters
        ----------
        source_dir : str
            Directory where the OpenMC model was exported (contains XML files).
        cache_dir : str
            Directory to store the cached copies.

        Returns
        -------
        list[str]
            Paths to the cached files.
        """
        shared, _ = self.shared_xml_files()
        os.makedirs(cache_dir, exist_ok=True)

        cached = []
        for xml_name in sorted(shared):
            src = os.path.join(source_dir, xml_name)
            dst = os.path.join(cache_dir, xml_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                cached.append(dst)
        return cached

    def restore_shared_xml(self, cache_dir, target_dir):
        """
        Restore cached shared XML files into *target_dir*.

        Call this before each subsequent UQ run so that invariant XML
        files do not need to be regenerated.

        Parameters
        ----------
        cache_dir : str
            Directory containing the cached XML files.
        target_dir : str
            Working directory for the current run.

        Returns
        -------
        list[str]
            Paths to the restored files.
        """
        shared, _ = self.shared_xml_files()
        os.makedirs(target_dir, exist_ok=True)

        restored = []
        for xml_name in sorted(shared):
            src = os.path.join(cache_dir, xml_name)
            dst = os.path.join(target_dir, xml_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                restored.append(dst)
        return restored

    # -----------------------------------------------------------------
    # Introspection helpers
    # -----------------------------------------------------------------

    @property
    def base_config_path(self):
        """Absolute path to the base configuration file."""
        return self._base_config_path

    @property
    def parameters(self):
        """The uncertain-parameter specs (read-only copy)."""
        return dict(self._parameters)

    @property
    def parameter_names(self):
        """Short names of all uncertain parameters."""
        return list(self._parameters.keys())

    def __repr__(self):
        return (
            f"ParametricModelBuilder("
            f"base={os.path.basename(self._base_config_path)!r}, "
            f"params={self.parameter_names})"
        )
