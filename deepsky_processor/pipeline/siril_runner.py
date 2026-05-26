"""Thin wrapper around the real Siril command-line tool."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path


def resolve_siril_cli(command: str = "siril-cli") -> str:
    """Return the configured Siril CLI executable or raise a clear error."""

    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Siril CLI executable was not found: {command}. "
            "Install Siril or set SIRIL_CLI to the executable path."
        )
    return resolved


def build_siril_command(
    command: str = "siril-cli",
    args: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Build a Siril command, optionally wrapped with xvfb-run."""

    args = args or []
    env = env or {}
    executable = resolve_siril_cli(command)

    if env.get("USE_XVFB") == "1":
        xvfb_run = shutil.which("xvfb-run")
        if xvfb_run is None:
            raise FileNotFoundError(
                "USE_XVFB=1 was set, but xvfb-run was not found on PATH. "
                "Install xvfb in the worker container or unset USE_XVFB."
            )
        return [xvfb_run, "-a", executable, *args]

    return [executable, *args]


def get_siril_version(command: str = "siril-cli", env: Mapping[str, str] | None = None) -> str:
    """Call the real Siril CLI and return its version output."""

    command_line = build_siril_command(command, ["--version"], env)
    result = subprocess.run(
        command_line,
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr).strip()


def run_siril_script(
    script_path: Path,
    working_directory: Path,
    command: str = "siril-cli",
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a real Siril script in a working directory."""

    script_path = script_path.expanduser().resolve()
    working_directory = working_directory.expanduser().resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Siril script does not exist: {script_path}")
    if not script_path.is_file():
        raise FileNotFoundError(f"Siril script path is not a file: {script_path}")
    if not working_directory.exists():
        raise FileNotFoundError(f"Siril working directory does not exist: {working_directory}")
    if not working_directory.is_dir():
        raise FileNotFoundError(
            f"Siril working directory is not a directory: {working_directory}"
        )

    command_line = build_siril_command(command, ["-s", str(script_path)], env)
    return subprocess.run(
        command_line,
        check=True,
        capture_output=True,
        cwd=working_directory,
        text=True,
    )
