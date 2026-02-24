#!/usr/bin/env python3
"""
openmc_model_run.py – EasyVVUQ wrapper for OpenMC TBM pebble-bed model.

Reads a YAML configuration file, builds and runs an OpenMC fixed-source
irradiation model of a Test Blanket Module (TBM) with Li2TiO3 ceramic pebbles,
then writes the tritium production rate to a CSV file that the EasyVVUQ
decoder can read.

Usage
-----
    python openmc_model_run.py --config config.yaml
"""

import argparse
import csv
import os
import sys

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
    Build an OpenMC TBM pebble-bed irradiation model from *config*.

    Geometry (from the problem statement notebook):
      * Rotating Li target wheel: layered disk (Li / Cu / H2O / Cu / vacuum /
        graphite / vacuum / Ti) inside a ZCylinder.
      * Test Blanket Module (TBM): Eurofer-97 casing filled with Li2TiO3
        ceramic – either as a solid monoblock or as randomly packed pebbles
        with air between them.
      * Room: air-filled box with vacuum boundary conditions.

    Run mode : fixed source (14.1 MeV DT-fusion neutrons in the Li target).
    QoIs     : tritium production rate (reactions per source neutron) in the
               Li2TiO3 ceramic; total neutron flux across all materials.
    """
    import openmc

    # ── Read config sections ─────────────────────────────────────────────────
    geom_cfg = config.get('geometry', {})
    mat_cfg  = config.get('materials', {})
    sim_cfg  = config.get('simulation', {})

    # ── Geometry parameters ──────────────────────────────────────────────────
    # Target layer thicknesses (cm)
    li_thickness       = float(geom_cfg.get('li_thickness',       0.02))
    cu_thickness       = float(geom_cfg.get('cu_thickness',       0.3))
    water_thickness    = float(geom_cfg.get('water_thickness',    0.6))
    vacuum_thickness_1 = float(geom_cfg.get('vacuum_thickness_1', 1.5))
    graphite_thickness = float(_get_mean(geom_cfg.get('graphite_thickness', 0.7)))
    vacuum_thickness_2 = float(geom_cfg.get('vacuum_thickness_2', 0.48))
    ti_thickness       = float(geom_cfg.get('ti_thickness',       0.6))
    air_gap            = float(geom_cfg.get('air_gap',            0.1))
    wh_r               = float(geom_cfg.get('wh_r',               50.0))

    # TBM casing dimensions (cm)
    eurofer_thickness = float(geom_cfg.get('eurofer_thickness', 0.5))
    tbm_width         = float(geom_cfg.get('tbm_width',         7.0))
    tbm_thickness     = float(geom_cfg.get('tbm_thickness',     2.0))
    tbm_height        = float(geom_cfg.get('tbm_height',        3.0))
    tbm_position_y    = float(geom_cfg.get('tbm_position_y',    -42.0))

    # Pebble bed parameters (may be uncertain / UQ-varied)
    pebbles_or_monoblock = geom_cfg.get('pebbles_or_monoblock', 'pebbles')
    pebble_radius    = float(_get_mean(geom_cfg.get('pebble_radius',    0.1)))
    packing_fraction = float(_get_mean(geom_cfg.get('packing_fraction', 0.3)))

    # ── Material parameters ──────────────────────────────────────────────────
    li_ceramic_density = float(
        _get_mean(mat_cfg.get('li_ceramic', {}).get('density',        3.43)))
    li6_enrichment     = float(
        _get_mean(mat_cfg.get('li_ceramic', {}).get('li6_enrichment', 7.5)))

    # ── Simulation settings ──────────────────────────────────────────────────
    particles  = int(sim_cfg.get('particles', 10000))
    batches    = int(sim_cfg.get('batches',   100))
    seed       = int(sim_cfg.get('seed',      1))
    output_dir = sim_cfg.get('output_directory', 'results/')

    os.makedirs(output_dir, exist_ok=True)

    # ── Materials ────────────────────────────────────────────────────────────
    # Li2TiO3 ceramic breeder
    li_ceramic = openmc.Material(name='Li2TiO3')
    li_ceramic.add_element('Li', 2.0, percent_type='ao',
                           enrichment=li6_enrichment,
                           enrichment_target='Li6',
                           enrichment_type='wo')
    li_ceramic.add_element('Ti', 1.0, percent_type='ao')
    li_ceramic.add_element('O',  3.0, percent_type='ao')
    li_ceramic.set_density('g/cm3', li_ceramic_density)
    li_ceramic.depletable = True

    # Eurofer-97 RAFM steel
    eurofer_97 = openmc.Material(name='Eurofer97')
    eurofer_97.add_element('C',  0.11)
    eurofer_97.add_element('Cr', 9.0)
    eurofer_97.add_element('W',  1.1)
    eurofer_97.add_element('Mn', 0.4)
    eurofer_97.add_element('Ta', 0.12)
    eurofer_97.add_nuclide('N14', 0.03)
    eurofer_97.add_element('Fe', 91.24)
    eurofer_97.set_density('g/cm3', 7.798)
    eurofer_97.depletable = False

    # Natural lithium target
    li_target = openmc.Material(name='Lithium')
    li_target.add_element('Li', 1.0)
    li_target.set_density('g/cm3', 0.534)
    li_target.depletable = False

    # Copper
    cu = openmc.Material(name='Copper')
    cu.add_element('Cu', 1.0)
    cu.set_density('g/cm3', 8.96)
    cu.depletable = False

    # Titanium
    ti_mat = openmc.Material(name='Titanium')
    ti_mat.add_element('Ti', 1.0)
    ti_mat.set_density('g/cm3', 4.506)
    ti_mat.depletable = False

    # Water coolant with thermal scattering
    h2o = openmc.Material(name='Water')
    h2o.add_nuclide('H1',  2.0)
    h2o.add_nuclide('O16', 1.0)
    h2o.add_s_alpha_beta('c_H_in_H2O')
    h2o.set_density('g/cm3', 1.0)
    h2o.depletable = False

    # Graphite shielding
    graphite = openmc.Material(name='Graphite')
    graphite.add_element('C', 1.0)
    graphite.set_density('g/cm3', 2.1)
    graphite.depletable = False

    # Air
    air = openmc.Material(name='Air')
    air.add_element('N',  0.78)
    air.add_element('O',  0.21)
    air.add_element('Ar', 0.01)
    air.set_density('g/cm3', 1.225e-3)
    air.depletable = False

    materials = openmc.Materials(
        [li_ceramic, li_target, cu, ti_mat, h2o, graphite, air, eurofer_97])

    # ── Surfaces ─────────────────────────────────────────────────────────────
    # Target wheel cylinder (z-axis)
    cylinder = openmc.ZCylinder(r=wh_r)

    # Z-planes for target layer boundaries
    z_li_lo    = -li_thickness / 2
    z_li_hi    =  li_thickness / 2
    z_cu1_hi   = z_li_hi + cu_thickness
    z_water_hi = z_cu1_hi + water_thickness
    z_cu2_hi   = z_water_hi + cu_thickness
    z_vac1_hi  = z_cu2_hi + vacuum_thickness_1
    z_graph_hi = z_vac1_hi + graphite_thickness
    z_vac2_hi  = z_graph_hi + vacuum_thickness_2
    z_ti_hi    = z_vac2_hi + ti_thickness

    li_start   = openmc.ZPlane(z0=z_li_lo)
    li_end     = openmc.ZPlane(z0=z_li_hi)
    cu_1_end   = openmc.ZPlane(z0=z_cu1_hi)
    water_end  = openmc.ZPlane(z0=z_water_hi)
    cu_2_end   = openmc.ZPlane(z0=z_cu2_hi)
    vac_1_end  = openmc.ZPlane(z0=z_vac1_hi)
    graph_end  = openmc.ZPlane(z0=z_graph_hi)
    vac_2_end  = openmc.ZPlane(z0=z_vac2_hi)
    ti_end     = openmc.ZPlane(z0=z_ti_hi)

    # TBM z-offset (starts after air gap)
    base_case = z_ti_hi + air_gap

    # TBM casing surfaces
    casing_start  = openmc.ZPlane(z0=base_case)
    casing_end    = openmc.ZPlane(z0=base_case + tbm_thickness)
    casing_left   = openmc.XPlane(x0=-(tbm_width / 2 + eurofer_thickness))
    casing_right  = openmc.XPlane(x0= (tbm_width / 2 + eurofer_thickness))
    casing_bottom = openmc.YPlane(y0=tbm_position_y - (tbm_height / 2 + eurofer_thickness))
    casing_top    = openmc.YPlane(y0=tbm_position_y + (tbm_height / 2 + eurofer_thickness))

    # TBM inner (ceramic) surfaces
    inner_start  = openmc.ZPlane(z0=base_case + eurofer_thickness)
    inner_end    = openmc.ZPlane(z0=base_case + tbm_thickness - eurofer_thickness)
    inner_left   = openmc.XPlane(x0=-tbm_width / 2)
    inner_right  = openmc.XPlane(x0= tbm_width / 2)
    inner_bottom = openmc.YPlane(y0=tbm_position_y - tbm_height / 2)
    inner_top    = openmc.YPlane(y0=tbm_position_y + tbm_height / 2)

    # Room boundary surfaces (vacuum boundary conditions)
    back_wall  = openmc.ZPlane(z0=-50.0,  boundary_type='vacuum')
    front_wall = openmc.ZPlane(z0=250.0,  boundary_type='vacuum')
    left_wall  = openmc.XPlane(x0=-150.0, boundary_type='vacuum')
    right_wall = openmc.XPlane(x0= 150.0, boundary_type='vacuum')
    floor      = openmc.YPlane(y0=-100.0, boundary_type='vacuum')
    ceiling    = openmc.YPlane(y0= 100.0, boundary_type='vacuum')

    # ── Cells ─────────────────────────────────────────────────────────────────
    # Target layer cells (inside wheel cylinder, in target z-range)
    li_target_cell  = openmc.Cell(name='Li_cell',      fill=li_target,
                                  region=-cylinder & +li_start  & -li_end)
    cu_cell_1       = openmc.Cell(name='Cu_cell_1',    fill=cu,
                                  region=-cylinder & +li_end     & -cu_1_end)
    water_cell      = openmc.Cell(name='water_cell',   fill=h2o,
                                  region=-cylinder & +cu_1_end   & -water_end)
    cu_cell_2       = openmc.Cell(name='Cu_cell_2',    fill=cu,
                                  region=-cylinder & +water_end  & -cu_2_end)
    vac_cell_1      = openmc.Cell(name='vac_cell_1',   fill=None,
                                  region=-cylinder & +cu_2_end   & -vac_1_end)
    graphite_cell   = openmc.Cell(name='graphite_cell',fill=graphite,
                                  region=-cylinder & +vac_1_end  & -graph_end)
    vac_cell_2      = openmc.Cell(name='vac_cell_2',   fill=None,
                                  region=-cylinder & +graph_end  & -vac_2_end)
    ti_cell         = openmc.Cell(name='ti_cell',      fill=ti_mat,
                                  region=-cylinder & +vac_2_end  & -ti_end)

    # TBM regions
    inner_region  = (+inner_left & -inner_right & +inner_bottom & -inner_top
                     & +inner_start & -inner_end)
    casing_region = (+casing_left & -casing_right & +casing_bottom & -casing_top
                     & +casing_start & -casing_end & ~inner_region)
    casing_cell = openmc.Cell(name='casing_cell', fill=eurofer_97,
                              region=casing_region)

    # The "wheel_region" is the target layer region inside the cylinder.
    # This is excluded from the room so that the room only fills outside the
    # cylinder target layers (and outside the TBM).
    wheel_region = -cylinder & +li_start & -ti_end

    # Room cell: air everywhere outside the wheel target layers and TBM
    room_region = (+back_wall & -front_wall & +left_wall & -right_wall
                   & +floor & -ceiling
                   & ~wheel_region & ~casing_region & ~inner_region)
    room_cell = openmc.Cell(name='room_cell', fill=air, region=room_region)

    # Base cell list (target cells, casing, room – without TBM inner filling)
    base_cells = [
        li_target_cell, cu_cell_1, water_cell, cu_cell_2,
        vac_cell_1, graphite_cell, vac_cell_2, ti_cell,
        casing_cell, room_cell,
    ]

    # ── TBM ceramic fill ──────────────────────────────────────────────────────
    if pebbles_or_monoblock == 'monoblock':
        inner_cell = openmc.Cell(name='inner_cell', fill=li_ceramic,
                                 region=inner_region)
        all_cells = base_cells + [inner_cell]

    else:  # pebble bed
        sphere_locations = openmc.model.pack_spheres(
            region=inner_region,
            radius=pebble_radius,
            pf=packing_fraction,
        )
        print(f"Packed {len(sphere_locations)} pebbles "
              f"(r={pebble_radius} cm, pf={packing_fraction})")

        # One Li2TiO3 cell per pebble (inside sphere)
        pebble_surfaces = []
        pebble_cells = []
        for loc in sphere_locations:
            sph = openmc.Sphere(x0=float(loc[0]), y0=float(loc[1]),
                                z0=float(loc[2]), r=pebble_radius)
            pebble_surfaces.append(sph)
            pebble_cells.append(openmc.Cell(fill=li_ceramic, region=-sph))

        # Air fills the space between pebbles inside the TBM
        air_between = inner_region
        for sph in pebble_surfaces:
            air_between = air_between & (+sph)
        inner_cell = openmc.Cell(name='inner_cell', fill=air, region=air_between)

        all_cells = base_cells + pebble_cells + [inner_cell]

    geometry = openmc.Geometry(all_cells)

    # ── Settings ─────────────────────────────────────────────────────────────
    settings = openmc.Settings()
    settings.batches   = batches
    settings.particles = particles
    settings.seed      = seed
    settings.run_mode  = 'fixed source'

    # 14.1 MeV isotropic neutron source inside the Li target disk
    source_lo = [-(wh_r - 1.0), -(wh_r - 1.0), z_li_lo]
    source_hi = [+(wh_r - 1.0), +(wh_r - 1.0), z_li_hi]
    space  = openmc.stats.Box(source_lo, source_hi)
    energy = openmc.stats.Discrete([14.1e6], [1.0])
    settings.source = openmc.IndependentSource(space=space, energy=energy)

    # ── Tallies ───────────────────────────────────────────────────────────────
    # Score (n,t) reactions in the Li2TiO3 ceramic to get TBR per source neutron
    tally = openmc.Tally(name='tritium_production')
    tally.filters = [openmc.MaterialFilter([li_ceramic])]
    tally.scores  = ['(n,t)']

    # Score total neutron flux across all materials
    flux_tally = openmc.Tally(name='total_neutron_flux')
    flux_tally.scores = ['flux']

    tallies = openmc.Tallies([tally, flux_tally])

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
        Names of QoIs requested in the config.
        Supported: ``'tritium_production_rate'``, ``'total_neutron_flux'``, ``'k_eff'``.

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

        if 'tritium_production_rate' in qoi_names:
            try:
                tally = sp.get_tally(name='tritium_production')
                df  = tally.get_pandas_dataframe()
                tpr = float(df['mean'].sum())
                results['tritium_production_rate'] = tpr
                print(f"  tritium_production_rate = {tpr:.4e}")
            except Exception as exc:
                print(f"  Warning: could not extract tritium_production_rate: {exc}")
                results['tritium_production_rate'] = 0.0

        if 'total_neutron_flux' in qoi_names:
            try:
                tally = sp.get_tally(name='total_neutron_flux')
                df  = tally.get_pandas_dataframe()
                flux = float(df['mean'].sum())
                results['total_neutron_flux'] = flux
                print(f"  total_neutron_flux = {flux:.4e}")
            except Exception as exc:
                print(f"  Warning: could not extract total_neutron_flux: {exc}")
                results['total_neutron_flux'] = 0.0

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
    print("\n ! Entering OpenMC EasyVVUQ model wrapper (TBM pebble-bed) !\n")

    parser = argparse.ArgumentParser(
        description='Run OpenMC TBM pebble-bed model with YAML configuration'
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
