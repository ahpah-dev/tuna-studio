"""Keep direct CLI launches on this app's configured Python environment."""
from pathlib import Path
import os
import subprocess
import sys


def terminate_worker(process, timeout=8):
    """Stop the interpreter and its children, including Windows venv launchers."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Terminating just Scripts/python.exe can leave its interpreter child
        # holding CUDA memory and the stdout pipe open indefinitely.
        try:
            result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        process.terminate()
    except OSError:
        if process.poll() is None:
            raise


def reap_worker(process, timeout=8):
    """Bound cancellation and escalation rather than leaving a stuck worker."""
    terminate_worker(process, timeout=timeout)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_worker(process, timeout=timeout)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=timeout)


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
