"""Configuration checks for the real StarNet++ executable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def validate_starnet_path(path: Path | None, require_executable: bool = False) -> Path:
    """Validate that StarNet++ is configured and points to a real file."""

    if path is None:
        raise FileNotFoundError(
            "StarNet++ path is not configured. Set STARNET_PATH to the "
            "StarNet++ executable."
        )
    expanded = path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"StarNet++ executable does not exist: {expanded}")
    if not expanded.is_file():
        raise FileNotFoundError(f"StarNet++ path is not a file: {expanded}")
    if require_executable:
        import os

        if not os.access(expanded, os.X_OK):
            raise PermissionError(f"StarNet++ path is not executable: {expanded}")
    return expanded


def run_starnet(
    input_path: Path,
    output_path: Path,
    starnet_path: Path | None,
    args_template: list[str] | None = None,
    require_executable: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run the real StarNet++ executable."""

    executable = validate_starnet_path(starnet_path, require_executable=require_executable)
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"StarNet++ input image does not exist: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"StarNet++ input path is not a file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    args_template = args_template or ["{input}", "{output}", "256"]
    args = [
        arg.format(input=str(input_path), output=str(output_path))
        for arg in args_template
    ]
    env = _starnet_env(executable)

    result = subprocess.run(
        [str(executable), *args],
        check=True,
        capture_output=True,
        cwd=executable.parent,
        env=env,
        text=True,
    )
    if not output_path.exists():
        raise RuntimeError(
            f"StarNet++ completed but did not create expected output: {output_path}"
        )
    return result


def get_starnet_help(
    starnet_path: Path | None,
    require_executable: bool = False,
) -> str:
    """Call the real StarNet++ executable with --help or no args."""

    executable = validate_starnet_path(starnet_path, require_executable=require_executable)
    env = _starnet_env(executable)
    for args in (["--help"], []):
        result = subprocess.run(
            [str(executable), *args],
            capture_output=True,
            cwd=executable.parent,
            env=env,
            text=True,
        )
        output = (result.stdout or result.stderr).strip()
        if output:
            return output
    return "StarNet++ executable ran but did not print help text."


def _starnet_env(executable: Path) -> dict[str, str]:
    env = os.environ.copy()
    directory = str(executable.parent)
    current = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = f"{directory}:{current}" if current else directory
    return env
