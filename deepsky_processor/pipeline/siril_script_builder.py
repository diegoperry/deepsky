"""Generate Siril scripts for DeepSky jobs."""

from __future__ import annotations

from pathlib import Path

from deepsky_processor.pipeline.job_layout import JobLayout


def build_preprocess_script(layout: JobLayout, profile: str = "check") -> str:
    """Build a Siril preprocessing script for a job."""

    if profile == "check":
        return _build_check_script(layout)
    if profile == "osc-full":
        return _build_osc_full_script(layout)
    if profile == "lights-only":
        return _build_lights_only_script(layout)
    raise ValueError(f"Unknown Siril preprocessing profile: {profile}")


def _build_check_script(layout: JobLayout) -> str:
    """Build a conservative Siril script that only verifies script execution."""

    job_root = _siril_path(layout.root)
    output = _siril_path(layout.output)
    return "\n".join(
        [
            "# DeepSky Siril preprocessing script",
            "# Generated for the current job layout.",
            "# Profile: check",
            "# This script verifies Siril can execute against this job.",
            f'cd "{job_root}"',
            f"# lights: {_siril_path(layout.lights)}",
            f"# darks: {_siril_path(layout.darks)}",
            f"# flats: {_siril_path(layout.flats)}",
            f"# biases: {_siril_path(layout.biases)}",
            f"# output: {output}",
            "",
        ]
    )


def _build_osc_full_script(layout: JobLayout) -> str:
    """Build a full OSC calibration, registration, and stacking script."""

    job_root = _siril_path(layout.root)
    return "\n".join(
        [
            "# DeepSky Siril preprocessing script",
            "# Profile: osc-full",
            "# Expects RAW/FITS frames in lights, darks, flats, and biases.",
            "requires 1.2.0",
            f'cd "{job_root}"',
            "cd biases",
            "convert bias -out=../process",
            "cd ../process",
            "stack bias rej 3 3 -nonorm -out=../masters/bias_stacked",
            "cd ..",
            "cd flats",
            "convert flat -out=../process",
            "cd ../process",
            "calibrate flat -bias=../masters/bias_stacked",
            "stack pp_flat rej 3 3 -norm=mul -out=../masters/pp_flat_stacked",
            "cd ..",
            "cd darks",
            "convert dark -out=../process",
            "cd ../process",
            "stack dark rej 3 3 -nonorm -out=../masters/dark_stacked",
            "cd ..",
            "cd lights",
            "convert light -out=../process",
            "cd ../process",
            "calibrate light -dark=../masters/dark_stacked -flat=../masters/pp_flat_stacked -cc=dark -cfa -equalize_cfa -debayer",
            "register pp_light",
            "stack r_pp_light rej 3 3 -norm=addscale -out=../output/result",
            "cd ..",
            "",
        ]
    )


def _build_lights_only_script(layout: JobLayout) -> str:
    """Build a lights-only registration and stacking script."""

    job_root = _siril_path(layout.root)
    return "\n".join(
        [
            "# DeepSky Siril preprocessing script",
            "# Profile: lights-only",
            "# Expects RAW/FITS frames in lights. No calibration frames are used.",
            "requires 1.2.0",
            f'cd "{job_root}"',
            "cd lights",
            "convert light -out=../process",
            "cd ../process",
            "register light",
            "stack r_light rej 3 3 -norm=addscale -out=../output/result",
            "cd ..",
            "",
        ]
    )


def write_preprocess_script(layout: JobLayout, profile: str = "check") -> Path:
    """Write a generated Siril preprocessing script and return its path."""

    layout.scripts.mkdir(parents=True, exist_ok=True)
    layout.siril_script.write_text(build_preprocess_script(layout, profile), encoding="utf-8")
    return layout.siril_script


def _siril_path(path: Path) -> str:
    return path.as_posix()
