#!/usr/bin/env python3
"""
openmc_model_run.py – EasyVVUQ wrapper for OpenMC neutronics models.

Reads a YAML configuration file, builds and runs an OpenMC pin-cell model,
then writes the requested quantities of interest (QoIs) to a CSV file that
the EasyVVUQ decoder can read.

Usage
-----
    python openmc_model_run.py --config config.yaml
"""

import argparse
import csv
import os
import sys

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_file):
    """Load configuration from a YAML file."""
    try:
        with open(config_file, 'r') as fh:
            config = yaml.safe_load(fh)
        print(f"Configuration loaded from: {config_file}")
        return config
    except FileNotFoundError:
        print(f"Error: configuration file '{config_file}' not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None


# ---------------------------------------------------------------------------
# OpenMC model construction
# ---------------------------------------------------------------------------

def build_openmc_model(config):
    """
    Build an OpenMC pin-cell model from *config* and return the
    ``openmc.Model`` object.

    The geometry is a 2-D reflective square pin cell containing:
      * UO2 fuel pellet
      * Zircaloy cladding
      * Light-water moderator
    """
    import openmc

    # ── Geometry parameters ─────────────────────────────────────────────────
    geom_cfg = config.get('geometry', {})
    fuel_r = float(_get_mean(geom_cfg.get('fuel_radius', 0.41)))
    clad_ir = float(geom_cfg.get('cladding_inner_radius', 0.42))
    clad_or = float(geom_cfg.get('cladding_outer_radius', 0.475))
    pitch = float(geom_cfg.get('pitch', 1.26))

    # ── Material parameters ──────────────────────────────────────────────────
    mat_cfg = config.get('materials', {})

    fuel_density = float(_get_mean(mat_cfg.get('fuel', {}).get('density', 10.5)))
    enrichment = float(_get_mean(mat_cfg.get('fuel', {}).get('enrichment', 3.1)))
    clad_density = float(mat_cfg.get('cladding', {}).get('density', 6.55))
    mod_density = float(_get_mean(mat_cfg.get('moderator', {}).get('density', 0.7)))

    # ── Simulation settings ──────────────────────────────────────────────────
    sim_cfg = config.get('simulation', {})
    particles = int(sim_cfg.get('particles', 10000))
    batches = int(sim_cfg.get('batches', 150))
    inactive = int(sim_cfg.get('inactive', 50))
    seed = int(sim_cfg.get('seed', 1))
    output_dir = sim_cfg.get('output_directory', 'results/')

    os.makedirs(output_dir, exist_ok=True)

    # ── Materials ────────────────────────────────────────────────────────────
    fuel = openmc.Material(name='UO2 fuel')
    fuel.set_density('g/cm3', fuel_density)
    # enrichment is given as wt% U-235; the remainder is U-238
    u235_wo = enrichment / 100.0
    u238_wo = 1.0 - u235_wo
    fuel.add_nuclide('U235', u235_wo, 'wo')
    fuel.add_nuclide('U238', u238_wo, 'wo')
    fuel.add_element('O', 2 * (u235_wo / 235.0 + u238_wo / 238.0)
                     / ((u235_wo / 235.0 + u238_wo / 238.0)
                        + 2 * (u235_wo / 235.0 + u238_wo / 238.0)), 'ao')
    # Simpler approach: use the built-in enrichment helper
    fuel = openmc.Material(name='UO2 fuel')
    fuel.set_density('g/cm3', fuel_density)
    fuel.add_element('U', 1.0, enrichment=enrichment)
    fuel.add_element('O', 2.0)

    clad = openmc.Material(name='Zircaloy-4')
    clad.set_density('g/cm3', clad_density)
    clad.add_element('Zr', 0.98, 'wo')
    clad.add_element('Sn', 0.015, 'wo')
    clad.add_element('Fe', 0.002, 'wo')
    clad.add_element('Cr', 0.001, 'wo')
    clad.add_element('Ni', 0.0007, 'wo')

    water = openmc.Material(name='H2O moderator')
    water.set_density('g/cm3', mod_density)
    water.add_element('H', 2.0)
    water.add_element('O', 1.0)
    water.add_s_alpha_beta('c_H_in_H2O')

    materials = openmc.Materials([fuel, clad, water])

    # ── Geometry ─────────────────────────────────────────────────────────────
    fuel_cyl = openmc.ZCylinder(r=fuel_r)
    clad_inner = openmc.ZCylinder(r=clad_ir)
    clad_outer = openmc.ZCylinder(r=clad_or)

    half_pitch = pitch / 2.0
    left = openmc.XPlane(-half_pitch, boundary_type='reflective')
    right = openmc.XPlane(+half_pitch, boundary_type='reflective')
    bottom = openmc.YPlane(-half_pitch, boundary_type='reflective')
    top = openmc.YPlane(+half_pitch, boundary_type='reflective')

    fuel_region = -fuel_cyl
    gap_region = +fuel_cyl & -clad_inner
    clad_region = +clad_inner & -clad_outer
    mod_region = +clad_outer & +left & -right & +bottom & -top

    fuel_cell = openmc.Cell(fill=fuel, region=fuel_region, name='fuel')
    gap_cell = openmc.Cell(region=gap_region, name='gap')      # void gap
    clad_cell = openmc.Cell(fill=clad, region=clad_region, name='clad')
    mod_cell = openmc.Cell(fill=water, region=mod_region, name='moderator')

    universe = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, mod_cell])
    geometry = openmc.Geometry(universe)

    # ── Settings ─────────────────────────────────────────────────────────────
    settings = openmc.Settings()
    settings.batches = batches
    settings.inactive = inactive
    settings.particles = particles
    settings.seed = seed
    settings.run_mode = 'eigenvalue'

    # Uniform spatial source in the fuel
    bounds = [-fuel_r, -fuel_r, -1.0, fuel_r, fuel_r, 1.0]
    uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
    settings.source = openmc.IndependentSource(
        space=uniform_dist,
        constraints={'fissionable': True},
    )

    # ── Tallies ───────────────────────────────────────────────────────────────
    tallies = openmc.Tallies()

    # ── Assemble model ────────────────────────────────────────────────────────
    model = openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies,
    )

    return model, output_dir


# ---------------------------------------------------------------------------
# Results extraction
# ---------------------------------------------------------------------------

def extract_qois(statepoint_file, qoi_names):
    """
    Extract quantities of interest from an OpenMC statepoint file.

    Parameters
    ----------
    statepoint_file : str
        Path to the statepoint HDF5 file written by OpenMC.
    qoi_names : list[str]
        Names of QoIs requested in the config (currently only 'k_eff').

    Returns
    -------
    dict
        Mapping from QoI name to value.
    """
    import openmc

    results = {}
    with openmc.StatePoint(statepoint_file) as sp:
        if 'k_eff' in qoi_names:
            k_combined = sp.keff
            results['k_eff'] = float(k_combined.n)
            print(f"  k_eff = {k_combined}")

    return results


def save_results_csv(results, output_file):
    """Write QoI results to a CSV file for EasyVVUQ's SimpleCSV decoder."""
    fieldnames = list(results.keys())
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)
    print(f"Results saved to {output_file}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_mean(value):
    """
    Return the mean/nominal value regardless of whether *value* is a plain
    scalar or a mapping that contains a 'mean' key (UQ-spec format).
    """
    if isinstance(value, dict):
        return value.get('mean', 0.0)
    return value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n ! Entering OpenMC EasyVVUQ model wrapper !\n")

    parser = argparse.ArgumentParser(
        description='Run OpenMC pin-cell model with YAML configuration'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)',
    )
    args = parser.parse_args()

    print(f"Python executable : {sys.executable}")
    print(f"Configuration file: {args.config}")
    print(f"Working directory : {os.getcwd()}")

    config = load_config(args.config)
    if config is None:
        print("Failed to load configuration – aborting.")
        sys.exit(1)

    # Build model
    model, output_dir = build_openmc_model(config)

    # Run OpenMC (writes statepoint.*.h5 to the working directory)
    last_batch = config.get('simulation', {}).get('batches', 150)
    statepoint_file = f"statepoint.{last_batch:d}.h5"

    model.run()

    # Extract QoIs
    qoi_names = config.get('output', {}).get('qoi', ['k_eff'])
    results = extract_qois(statepoint_file, qoi_names)

    # Write CSV for EasyVVUQ decoder
    results_file = config.get('output', {}).get('results_file', 'results.csv')
    save_results_csv(results, results_file)

    print("OpenMC simulation completed successfully!")
    return results


if __name__ == "__main__":
    main()
