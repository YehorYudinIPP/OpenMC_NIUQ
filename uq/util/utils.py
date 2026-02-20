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


def validate_execution_setup():
    """Validate that the execution environment is properly configured."""
    runnable_script = "openmc_model_run.py"
    script_path = os.path.join(os.getcwd(), runnable_script)

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    if not os.access(script_path, os.X_OK):
        print(f"Making script executable: {script_path}")
        os.chmod(script_path, 0o755)

    python_exe = get_openmc_python()
    if not os.path.exists(python_exe):
        raise FileNotFoundError(f"Python executable not found: {python_exe}")

    print(f"✓ Script validation passed: {script_path}")
    print(f"✓ Python executable: {python_exe}")
    return python_exe, script_path


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
