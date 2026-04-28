#!/bin/bash
# =============================================================================
# ARCHER2 SLURM batch script – OpenMC UQ campaign (1 full node, 128 cores)
# =============================================================================
#
# ARCHER2 compute node: 2× AMD EPYC 7742 = 128 cores / node
#
# Submission:
#   sbatch slurm/archer2_openmc_uq.sh
#
# Customise every line marked  <<<  before submitting.
# =============================================================================

#SBATCH --job-name=openmc_uq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=<YOUR_BUDGET_CODE>     # <<< replace with your ARCHER2 budget code

# =============================================================================
# ARCHER2 filesystem paths
# =============================================================================
# $WORK is the high-performance Lustre filesystem on ARCHER2.
# It is set automatically in interactive sessions but must be fixed explicitly
# inside a batch job so that child processes (QCG-PJ tasks) inherit the
# correct value.
#
# Pattern:  /work/<project_code>/<group_code>/<username>
# Example:  /work/e723/e723/jsmith
export WORK="/work/YOUR_PROJECT/YOUR_GROUP/YOUR_USERNAME"  # <<< edit this

# =============================================================================
# Repository and Python-environment paths
# =============================================================================
# Directory where this repository is checked out on the Lustre filesystem:
REPO_DIR="$WORK/OpenMC_NIUQ"                                   # <<< edit if different

# Python virtualenv that contains EasyVVUQ, qcg-pilotjob, and easyqcgpj.
# This must NOT be the Spack OpenMC environment — it is the orchestration env.
# Create it once with:
#   python3 -m venv "$WORK/easyvvuq-env"
#   source "$WORK/easyvvuq-env/bin/activate"
#   pip install easyvvuq qcg-pilotjob easyqcgpj chaospy numpy pyyaml matplotlib
VENV_DIR="$WORK/easyvvuq-env"                                  # <<< edit if different

# =============================================================================
# Spack / OpenMC settings (ARCHER2-specific)
# =============================================================================
# Full path to the shared Spack installation on ARCHER2 – do not change:
SPACK_BIN="/mnt/lustre/a2fs-nvme/work/y07/shared/apps/dev/spack/1.0.2/spack/bin/spack"

# Name of the Spack environment that contains OpenMC.
# Create it once with:
#   module load other-software
#   module load spack
#   spack env create my_openmc_env
#   spack env activate my_openmc_env
#   spack install openmc
OPENMC_SPACK_ENV="my_openmc_env"                               # <<< edit if you used a different name

# =============================================================================
# UQ campaign settings
# =============================================================================
UQ_SCHEME="pce"     # pce  – Polynomial Chaos Expansion
                    # qmc  – Quasi-Monte Carlo
P_ORDER=2           # polynomial order for PCE  (ignored for qmc)
N_SAMPLES=256       # number of samples for QMC (ignored for pce)
CONFIG_FILE="$REPO_DIR/uq/config/model_config.yaml"  # <<< edit if needed

# =============================================================================
# Environment setup
# =============================================================================
cd "$REPO_DIR" || { echo "ERROR: repo dir not found: $REPO_DIR"; exit 1; }

# Load the ARCHER2 module environment for Spack.
# This sets SPACK_ROOT, SPACK_SYSTEM_CONFIG_PATH, and similar variables that
# allow Spack to locate named environments (like $OPENMC_SPACK_ENV) even when
# called from a subprocess later.
#
# These variables ARE inherited by QCG-PJ worker processes, so loading the
# modules here is sufficient – there is no need to reload them inside each
# individual OpenMC task.
module load other-software
module load spack

# Activate the EasyVVUQ/QCG-PJ orchestration virtualenv.
# (The Spack OpenMC environment is NOT activated here; each individual sample
#  task is wrapped with  "$SPACK_BIN -e $OPENMC_SPACK_ENV env run python3 ..."
#  which activates the Spack env in-process for that task only.)
source "$VENV_DIR/bin/activate"

# =============================================================================
# Pre-flight checks
# =============================================================================
echo "============================================================"
echo "  OpenMC UQ campaign – ARCHER2"
echo "============================================================"
echo "Job ID           : $SLURM_JOB_ID"
echo "Node list        : $SLURM_JOB_NODELIST"
echo "Tasks allocated  : $SLURM_NTASKS"
echo "WORK             : $WORK"
echo "Repo dir         : $REPO_DIR"
echo "Virtualenv       : $VENV_DIR"
echo "Spack binary     : $SPACK_BIN"
echo "OpenMC Spack env : $OPENMC_SPACK_ENV"
echo "UQ scheme        : $UQ_SCHEME  (p_order=$P_ORDER  n_samples=$N_SAMPLES)"
echo "Config file      : $CONFIG_FILE"
echo "------------------------------------------------------------"

# Verify that the Spack binary exists:
[[ -x "$SPACK_BIN" ]] || {
    echo "ERROR: Spack binary not found or not executable: $SPACK_BIN"
    exit 1
}

# Verify the named Spack environment exists:
"$SPACK_BIN" env list 2>/dev/null | grep -qw "$OPENMC_SPACK_ENV" || {
    echo "ERROR: Spack environment '$OPENMC_SPACK_ENV' not found."
    echo "       Create it with:"
    echo "         module load other-software && module load spack"
    echo "         spack env create $OPENMC_SPACK_ENV"
    echo "         spack env activate $OPENMC_SPACK_ENV"
    echo "         spack install openmc"
    exit 1
}

# Verify the Python virtualenv is usable:
[[ -f "$VENV_DIR/bin/activate" ]] || {
    echo "ERROR: Python virtualenv not found: $VENV_DIR"
    echo "       Create it with:"
    echo "         python3 -m venv $VENV_DIR"
    echo "         source $VENV_DIR/bin/activate"
    echo "         pip install easyvvuq qcg-pilotjob easyqcgpj chaospy numpy pyyaml matplotlib"
    exit 1
}

echo "Pre-flight checks passed."
echo "------------------------------------------------------------"
echo ""

# =============================================================================
# Launch EasyVVUQ campaign
# =============================================================================
#
# What happens at runtime
# -----------------------
# 1. This script runs the EasyVVUQ orchestrator  (single Python process).
# 2. The orchestrator calls QCG Pilot Job Manager (QCGPJPool / LocalManager).
# 3. QCG-PJ reads $SLURM_NTASKS (128) and distributes tasks across all cores.
# 4. For each UQ sample QCG-PJ spawns a subprocess running the exact command:
#
#      "$SPACK_BIN" -e "$OPENMC_SPACK_ENV" env run \
#          python3 openmc_model_run.py --config config.yaml
#
#    "spack -e <env> env run <cmd>" activates the named Spack environment
#    for that child process (sets PATH, LD_LIBRARY_PATH, etc.) without
#    modifying the parent shell.
# 5. Because $SPACK_ROOT and related variables were set above by
#    "module load spack", Spack can resolve "$OPENMC_SPACK_ENV" by name
#    even inside a subprocess that never ran "module load" itself.

python uq/easyvvuq_openmc.py \
    --exec-mode  spack              \
    --spack-bin  "$SPACK_BIN"       \
    --spack-env  "$OPENMC_SPACK_ENV" \
    --config     "$CONFIG_FILE"     \
    --uq-scheme  "$UQ_SCHEME"       \
    --p-order    "$P_ORDER"

echo ""
echo "Campaign finished.  Exit code: $?"
