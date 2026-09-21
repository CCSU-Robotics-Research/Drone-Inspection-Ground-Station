"""Launcher for the ground station.

Starts the gimbal teleoperation and camera vision components
with their respective venvs together. This is for production to
avoid manually running 2 terminals. However each component can
still be individually tested.

Ctrl+C shuts down the entire system. Additionally stopping the
gimbal teleoperation or camera vision stops all components.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Component name, directory under the repo root, and arguments for
# their main.py files.
_COMPONENTS = [
    ("GIMBAL", "gimbal_teleoperation", []),
    ("CAMERA", "camera_vision", []),
]


def _venv_python(component_dir: Path) -> Path:
    """Path to component's own venv interpreter."""
    if os.name == "nt":
        return component_dir / ".venv" / "Scripts" / "python.exe"
    raise OSError("You must use this on a Windows system.")


def _log_outputs(name: str, proc: subprocess.Popen) -> None:
    """Prefix and forward one component's output for logging."""
    for line in proc.stdout:
        print(f"[{name}] {line}", end="", flush=True)


def _start(name: str, directory: str, args: list) -> subprocess.Popen:
    """Start one component in its own directory with its own venv."""
    component_dir = _ROOT / directory
    interpreter = _venv_python(component_dir)
    if not interpreter.exists():
        sys.exit(
            f"[LAUNCH] no venv for {name} at {interpreter}\n"
            f"[LAUNCH] create it per {directory}/README.md first"
        )

    proc = subprocess.Popen(
        [str(interpreter), "main.py", *args],
        cwd=str(component_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    threading.Thread(
        target=_log_outputs, args=(name, proc), daemon=True
    ).start()
    print(f"[LAUNCH] {name} started, PID {proc.pid}")
    return proc


def _request_shutdown(procs: list) -> None:
    """Shut down all subprocesses."""
    os.kill(0, signal.CTRL_C_EVENT)


def _wait_for_exit(procs: list) -> None:
    """Delay in exit to allow for cleanup."""
    max_delay = time.monotonic() + 5.0
    for name, proc in procs:
        while True:
            remaining = max(0.5, max_delay - time.monotonic())
            try:
                proc.wait(timeout=remaining)
                break
            except subprocess.TimeoutExpired:
                print(f"[LAUNCH] {name} hasn't exited; force exiting...")
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break


def main() -> int:
    procs = []
    for name, directory, args in _COMPONENTS:
        procs.append((name, _start(name, directory, args)))

    print("[LAUNCH] all components launched. Ctrl+C to quit")

    user_interrupted = False
    try:
        while True:
            time.sleep(0.25)
            exited = [
                (name, proc) for name, proc in procs
                if proc.poll() is not None
            ]
            if exited:
                for name, proc in exited:
                    print(
                        f"[LAUNCH] {name} exited with code "
                        f"{proc.returncode}, stopping other processes..."
                    )
                break
    except KeyboardInterrupt:
        print("\n[LAUNCH] Ctrl+C: waiting for components to finish "
              "shutting down\n")
        user_interrupted = True
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if not user_interrupted:
            _request_shutdown(procs)
        _wait_for_exit(procs)

    worst_exit_code = 0
    for name, proc in procs:
        print(f"[LAUNCH] {name} exit code: {proc.returncode}")
        worst_exit_code = max(worst_exit_code, abs(proc.returncode or 0))
    print("[LAUNCH] done")
    return 0 if worst_exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
