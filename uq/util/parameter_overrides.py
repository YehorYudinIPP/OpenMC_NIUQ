"""
parameter_overrides.py – Utilities for parameter-only UQ configuration.

Provides reusable functions for:

* Loading a parameters-only YAML specification (no full model config).
* Merging uncertain-parameter overrides into a base configuration.
* Applying a dictionary of sampled values to a configuration.
* Identifying which OpenMC XML files are invariant across UQ runs so
  they can be generated once and shared.
"""

import copy
import os
import warnings

import yaml


# -- Section-to-XML mapping used by OpenMC ------------------------------------
_SECTION_XML_MAP = {
    "geometry":   "geometry.xml",
    "materials":  "materials.xml",
    "simulation": "settings.xml",
    "settings":   "settings.xml",
    "output":     "tallies.xml",
    "tallies":    "tallies.xml",
}

_ALL_XML_FILES = {"geometry.xml", "materials.xml", "settings.xml", "tallies.xml"}


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def load_uq_parameters(yaml_path):
    """
    Load a parameters-only YAML file.

    The expected format is::

        base_config: model_config.yaml

        parameters:
          graphite_thickness:
            path: geometry.graphite_thickness
            mean: 0.7
            relative_stdev: 0.05
            pdf: normal
          li_ceramic_density:
            path: materials.li_ceramic.density
            mean: 3.43
            relative_stdev: 0.02
            pdf: normal

    Parameters
    ----------
    yaml_path : str
        Path to the parameters-only YAML file.

    Returns
    -------
    dict
        Parsed YAML content with keys ``base_config`` and ``parameters``.

    Raises
    ------
    FileNotFoundError
        If *yaml_path* does not exist.
    ValueError
        If required keys are missing.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"UQ parameters file not found: {yaml_path}")

    with open(yaml_path, "r") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in '{yaml_path}', got {type(data).__name__}."
        )

    if "parameters" not in data:
        raise ValueError(
            f"UQ parameters file '{yaml_path}' must contain a 'parameters' key."
        )

    return data


def resolve_base_config_path(uq_params_path, base_config_rel):
    """
    Resolve the ``base_config`` path relative to the UQ parameters file.

    Parameters
    ----------
    uq_params_path : str
        Absolute or relative path to the UQ parameters YAML file.
    base_config_rel : str
        Value of the ``base_config`` key (may be relative to the YAML file).

    Returns
    -------
    str
        Absolute path to the base configuration file.
    """
    if os.path.isabs(base_config_rel):
        return base_config_rel
    uq_dir = os.path.dirname(os.path.abspath(uq_params_path))
    return os.path.normpath(os.path.join(uq_dir, base_config_rel))


# ---------------------------------------------------------------------------
# Deep-merge / set-nested helpers
# ---------------------------------------------------------------------------

def _set_nested(mapping, dot_path, value):
    """
    Set a value inside a nested dict using a dot-separated path.

    Intermediate dicts are created when they do not exist.

    Parameters
    ----------
    mapping : dict
        The dictionary to modify **in place**.
    dot_path : str
        Dot-separated key path, e.g. ``"materials.li_ceramic.density"``.
    value
        The value to set at the leaf.
    """
    keys = dot_path.split(".")
    current = mapping
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _get_nested(mapping, dot_path, default=None):
    """
    Retrieve a value from a nested dict using a dot-separated path.

    Parameters
    ----------
    mapping : dict
        The dictionary to read from.
    dot_path : str
        Dot-separated key path.
    default
        Returned when the path does not exist.

    Returns
    -------
    object
        The value at the path, or *default*.
    """
    keys = dot_path.split(".")
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# ---------------------------------------------------------------------------
# Merge parameters into a base config
# ---------------------------------------------------------------------------

def merge_parameters(base_config, uq_parameters):
    """
    Overlay uncertain-parameter specs onto a deep copy of *base_config*.

    Each entry in *uq_parameters* must have at least ``path`` and ``mean``.
    The value at the given YAML path is replaced with the full UQ-spec
    dictionary (``mean``, ``relative_stdev``, ``pdf``), which is the
    format expected by :func:`discover_uncertain_parameters` in
    ``easyvvuq_openmc.py``.

    Parameters
    ----------
    base_config : dict
        Full model configuration loaded from the base YAML file.
    uq_parameters : dict
        Mapping of parameter names to their specs.  Each spec must have
        ``path`` (dot-separated YAML path) and ``mean``.  Optional keys
        are ``relative_stdev`` (default 0.05) and ``pdf`` (default
        ``"uniform"``).

    Returns
    -------
    dict
        A **new** configuration dict with the uncertain parameters merged in.
    """
    merged = copy.deepcopy(base_config)

    for name, spec in uq_parameters.items():
        if "path" not in spec:
            warnings.warn(
                f"Parameter '{name}' is missing a 'path' key – skipping.",
                stacklevel=2,
            )
            continue

        uq_node = {
            "mean": spec.get("mean", 0.0),
            "relative_stdev": spec.get("relative_stdev", 0.05),
            "pdf": spec.get("pdf", "uniform"),
        }

        _set_nested(merged, spec["path"], uq_node)

    return merged


# ---------------------------------------------------------------------------
# Apply sampled values
# ---------------------------------------------------------------------------

def apply_sampled_values(config, sampled_values, parameter_specs):
    """
    Substitute sampled numeric values into a deep copy of *config*.

    This is used during the per-run construction phase to replace the
    ``mean`` field of each uncertain parameter with its sampled value.

    Parameters
    ----------
    config : dict
        Full model configuration (possibly with UQ-spec nodes).
    sampled_values : dict
        Mapping of parameter short names to sampled float values.
    parameter_specs : dict
        The ``parameters`` section of the UQ YAML (maps names to specs
        with at least a ``path`` key).

    Returns
    -------
    dict
        A **new** configuration dict with sampled values applied.
    """
    result = copy.deepcopy(config)

    for name, value in sampled_values.items():
        spec = parameter_specs.get(name)
        if spec is None:
            warnings.warn(
                f"Sampled parameter '{name}' not found in specs – skipping.",
                stacklevel=2,
            )
            continue

        dot_path = spec["path"]
        current = _get_nested(result, dot_path)

        if isinstance(current, dict) and "mean" in current:
            _set_nested(result, f"{dot_path}.mean", float(value))
        else:
            _set_nested(result, dot_path, float(value))

    return result


# ---------------------------------------------------------------------------
# Shared XML identification
# ---------------------------------------------------------------------------

def identify_shared_xml(parameter_specs):
    """
    Determine which OpenMC XML files are invariant across UQ runs.

    An XML file is shared (invariant) when **none** of the uncertain
    parameters belong to the corresponding model section.

    Parameters
    ----------
    parameter_specs : dict
        The ``parameters`` section of the UQ YAML.

    Returns
    -------
    tuple[set, set]
        ``(shared_xml, varying_xml)`` – two sets of XML file names.

    Examples
    --------
    >>> specs = {
    ...     "graphite_thickness": {"path": "geometry.graphite_thickness"},
    ... }
    >>> shared, varying = identify_shared_xml(specs)
    >>> "settings.xml" in shared
    True
    >>> "geometry.xml" in varying
    True
    """
    affected_sections = set()
    for spec in parameter_specs.values():
        path = spec.get("path", "")
        top_section = path.split(".")[0]
        affected_sections.add(top_section)

    varying_xml = set()
    for section in affected_sections:
        xml_file = _SECTION_XML_MAP.get(section)
        if xml_file:
            varying_xml.add(xml_file)

    shared_xml = _ALL_XML_FILES - varying_xml
    return shared_xml, varying_xml
