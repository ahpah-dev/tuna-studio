"""Keep direct CLI launches on this app's configured Python environment."""
from pathlib import Path
import os
import subprocess
import sys


def use_app_environment(script):
    app_dir = Path(script).resolve().parent
    environment = app_dir / ".venv"
    local_python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (local_python.is_file()
            and Path(sys.prefix).resolve() != environment.resolve()
            and os.environ.get("TUNA_USE_CURRENT_PYTHON") != "1"):
        # Waiting preserves stdout, exit codes and normal foreground CLI use on
        # Windows, where execv otherwise detaches from some terminal hosts.
        completed = subprocess.run([str(local_python), str(Path(script).resolve()), *sys.argv[1:]])
        raise SystemExit(completed.returncode)
