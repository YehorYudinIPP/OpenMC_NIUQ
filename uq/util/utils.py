import os
import subprocess
import sys
import yaml
from datetime import datetime


def load_config(config_file):
    """Load configuration from YAML file specific for UQ."""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        print(f" >> Configuration loaded from: {config_file}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def add_timestamp_to_filename(filename, timestamp=None):
    """
    Add timestamp to filename before the extension.

    Args:
        filename (str): Original filename
        timestamp (str, optional): Custom timestamp string. If None, uses current datetime.

    Returns:
        str: Filename with timestamp

    Example:
        add_timestamp_to_filename("results.csv") -> "results_20250718_143025.csv"
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"


def get_openmc_python():
    """Get the correct Python executable for the OpenMC environment."""

    # Method 1: Check for a dedicated conda/venv environment
    candidate_paths = [
        "/opt/openmc-env/bin/python3",
        os.path.expanduser("~/anaconda3/envs/openmc-env/bin/python3"),
        os.path.expanduser("~/miniconda3/envs/openmc-env/bin/python3"),
    ]
    for env_python in candidate_paths:
        if os.path.exists(env_python):
            return env_python

    # Method 2: Try to find an openmc conda environment dynamically
    try:
        result = subprocess.run(
            ['conda', 'info', '--envs'],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.split('\n'):
            if 'openmc' in line.lower():
                env_path = line.split()[-1]
                python_path = os.path.join(env_path, 'bin', 'python')
                if os.path.exists(python_path):
                    return python_path
    except Exception:
        pass

    # Method 3: Fallback to current Python
    print("Warning: OpenMC environment not found, using current Python")
    return sys.executable


# ---------------------------------------------------------------------------
# Execution-command builder
# ---------------------------------------------------------------------------

# ── Spack-related variables ───────────────────────────────────────────────
#
#  SPACK_ENV  (exec_params key: 'spack_env')
#      The Spack environment that contains OpenMC and all its runtime
#      libraries.  Can be either:
#        • A full filesystem path  – e.g. /work/e723/e723/jsmith/my_openmc_env
#        • A named environment     – e.g. my_openmc_env
#      Named environments are resolved by spack using the configuration scope
#      that was set up by "module load spack" (ARCHER2) or equivalent.
#      Default: $WORK/my_openc_env  (uses the $WORK env var when present).
#
#  SPACK_BIN  (exec_params key: 'spack_bin')
#      Full path (or bare name if on PATH) of the ``spack`` executable.
#      On ARCHER2 this is the shared installation; do NOT rely on the module-
#      provided PATH because QCG-PJ subprocesses may not inherit it reliably.
#      Default: first match among the known candidates, else ``spack``.
#      ARCHER2: /mnt/lustre/a2fs-nvme/work/y07/shared/apps/dev/spack/1.0.2/spack/bin/spack
#
#  The command issued for each sample in 'spack' mode is:
#      {SPACK_BIN} -e {SPACK_ENV} env run python3 {script_path} --config config.yaml
#
#  ``spack -e <env> env run <cmd>`` activates the named Spack environment
#  (sets PATH, LD_LIBRARY_PATH, etc.) for the child process without requiring
#  an interactive shell.  The EasyVVUQ / QCG-PJ orchestrator therefore stays
#  in the regular Python environment while every OpenMC instance runs inside
#  the Spack environment.
#
#  ARCHER2 note
#  ------------
#  On ARCHER2 the Spack module path is a shared installation.  Load the
#  modules in the SLURM batch script so that $SPACK_ROOT and related env vars
#  are inherited by QCG-PJ worker subprocesses:
#
#      module load other-software
#      module load spack
#      # then run the EasyVVUQ orchestrator with --exec-mode spack
#
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_SPACK_BIN_CANDIDATES = [
    # ARCHER2 shared Spack installation
    "/mnt/lustre/a2fs-nvme/work/y07/shared/apps/dev/spack/1.0.2/spack/bin/spack",
    # User-local installations
    os.path.expanduser("~/spack/bin/spack"),
    "/usr/local/bin/spack",
]


def _find_spack_bin():
    """Return the best available path to the ``spack`` executable."""
    for candidate in _DEFAULT_SPACK_BIN_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    # Fall back to bare name and rely on PATH
    return "spack"


def build_exec_command(script_path, exec_params):
    """
    Build the shell command that EasyVVUQ issues for each OpenMC sample.

    Parameters
    ----------
    script_path : str
        Absolute path to ``openmc_model_run.py``.
    exec_params : dict
        Controls how the command is built.  Recognised keys:

        exec_mode : {'local', 'spack'}
            * ``'local'``  – run with the current Python interpreter
              (conda / venv that has OpenMC installed).
            * ``'spack'``  – wrap with ``spack -e <env> env run python3``
              so that each OpenMC instance runs inside the Spack environment.
              Parallelism is managed by QCG-PJ (see ``run_uq_campaign``).

        spack_env : str, optional
            Spack environment containing OpenMC.  Accepts either:
              * A **named environment** (e.g. ``my_openmc_env``) – resolved by
                spack using the configuration scope set up by
                ``module load spack`` or equivalent.  This is the recommended
                form on ARCHER2.
              * A **full filesystem path** (e.g.
                ``/work/e723/e723/jsmith/my_openmc_env``).
            Default: ``$WORK/my_openc_env`` (falls back to
            ``~/my_openc_env`` when ``$WORK`` is not set).

        spack_bin : str, optional
            Path to the ``spack`` executable.
            Default: first match in the known-path list (includes the
            ARCHER2 shared installation), otherwise ``spack`` on PATH.
            **ARCHER2**: ``/mnt/lustre/a2fs-nvme/work/y07/shared/apps/dev/spack/1.0.2/spack/bin/spack``

    Returns
    -------
    str
        Full command string passed to ``ExecuteLocal`` / QCG-PJ for each run.

    Notes
    -----
    On ARCHER2 the Spack module must be loaded in the SLURM batch script
    *before* launching the EasyVVUQ orchestrator so that ``$SPACK_ROOT`` and
    related configuration variables are inherited by QCG-PJ subprocesses::

        module load other-software
        module load spack
        # then: python uq/easyvvuq_openmc.py --exec-mode spack ...

    The per-task command on ARCHER2 becomes::

        /mnt/lustre/.../spack -e my_openmc_env env run python3 openmc_model_run.py --config config.yaml
    """
    mode = exec_params.get('exec_mode', 'local')

    if mode == 'local':
        python_exe = get_openmc_python()
        return f"{python_exe} {script_path} --config config.yaml"

    if mode == 'spack':
        spack_env = exec_params.get(
            'spack_env',
            os.path.join(os.environ.get('WORK', os.path.expanduser('~')),
                         'my_openc_env'),
        )
        spack_bin = exec_params.get('spack_bin', _find_spack_bin())
        return (
            f"{spack_bin} -e {spack_env} env run "
            f"python3 {script_path} --config config.yaml"
        )

    raise ValueError(
        f"Unknown exec_mode '{mode}'. Choose: 'local' or 'spack'."
    )


def validate_execution_setup(exec_params=None):
    """
    Validate that the execution environment is properly configured and return
    the full execution command for a single OpenMC run.

    Parameters
    ----------
    exec_params : dict or None
        Passed directly to :func:`build_exec_command`.
        When *None*, defaults to ``{'exec_mode': 'local'}``.

    Returns
    -------
    exec_cmd : str
        The command string that will be issued for every sample run.
    script_path : str
        Absolute path to ``openmc_model_run.py``.
    """
    exec_params = exec_params or {'exec_mode': 'local'}

    runnable_script = "openmc_model_run.py"
    script_path = os.path.join(os.getcwd(), runnable_script)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    if not os.access(script_path, os.X_OK):
        print(f"Making script executable: {script_path}")
        os.chmod(script_path, 0o755)

    exec_cmd = build_exec_command(script_path, exec_params)

    print(f"✓ Script validation passed: {script_path}")
    print(f"✓ Execution command  : {exec_cmd}")
    return exec_cmd, script_path


def save_sa_results_yaml(results_dict, filename_base="sa_results.yaml"):
    """
    Save sensitivity analysis results to a YAML file.

    Args:
        results_dict (dict): Dictionary of results to save.
        filename_base (str): Base filename for the output file.

    Returns:
        str: Path to the saved file.
    """
    output = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': 'Sensitivity analysis results for OpenMC UQ campaign',
        'data': results_dict,
    }

    filename = add_timestamp_to_filename(filename_base)
    with open(filename, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, indent=2)

    print(f"✓ Sensitivity analysis results saved to: {filename}")
    return filename
