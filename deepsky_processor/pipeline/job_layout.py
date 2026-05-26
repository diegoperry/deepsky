"""Job directory layout helpers for DeepSky worker jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FRAME_DIRS = ("lights", "darks", "flats", "biases")
WORK_DIRS = ("process", "masters", "output", "scripts")
SUPPORTED_FRAME_EXTENSIONS = {
    ".fit",
    ".fits",
    ".fts",
    ".tif",
    ".tiff",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".dng",
    ".raf",
    ".rw2",
    ".orf",
}


@dataclass(frozen=True)
class JobLayout:
    root: Path
    lights: Path
    darks: Path
    flats: Path
    biases: Path
    output: Path
    process: Path
    masters: Path
    logs: Path
    scripts: Path
    siril_script: Path


def build_job_layout(job_dir: Path) -> JobLayout:
    """Return the expected DeepSky folder layout for a job."""

    root = job_dir.expanduser().resolve()
    scripts = root / "scripts"
    return JobLayout(
        root=root,
        lights=root / "lights",
        darks=root / "darks",
        flats=root / "flats",
        biases=root / "biases",
        output=root / "output",
        process=root / "process",
        masters=root / "masters",
        logs=root / "logs",
        scripts=scripts,
        siril_script=scripts / "siril_preprocess.ssf",
    )


def create_job_layout(job_dir: Path) -> JobLayout:
    """Create the expected DeepSky job folders."""

    layout = build_job_layout(job_dir)
    for directory in (
        layout.root,
        layout.lights,
        layout.darks,
        layout.flats,
        layout.biases,
        layout.process,
        layout.masters,
        layout.output,
        layout.logs,
        layout.scripts,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def validate_job_layout(job_dir: Path) -> JobLayout:
    """Validate that a job has the expected frame folders."""

    layout = build_job_layout(job_dir)
    if not layout.root.exists():
        raise FileNotFoundError(f"Job directory does not exist: {layout.root}")
    if not layout.root.is_dir():
        raise FileNotFoundError(f"Job path is not a directory: {layout.root}")

    for name in FRAME_DIRS:
        directory = getattr(layout, name)
        if not directory.exists():
            raise FileNotFoundError(f"Required job folder is missing: {directory}")
        if not directory.is_dir():
            raise FileNotFoundError(f"Required job path is not a directory: {directory}")
    return layout


def count_frame_files(directory: Path) -> int:
    """Count likely image frame files in a directory."""

    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_FRAME_EXTENSIONS
    )


def validate_profile_inputs(layout: JobLayout, profile: str) -> None:
    """Validate frame folders required by a generated Siril profile."""

    required = {
        "check": (),
        "lights-only": ("lights",),
        "osc-full": ("lights", "darks", "flats", "biases"),
    }.get(profile)
    if required is None:
        raise ValueError(f"Unknown Siril preprocessing profile: {profile}")

    missing = []
    for name in required:
        directory = getattr(layout, name)
        if count_frame_files(directory) == 0:
            missing.append(str(directory))

    if missing:
        raise FileNotFoundError(
            "No supported frame files found in required folder(s): "
            + ", ".join(missing)
        )
