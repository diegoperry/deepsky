"""Filesystem helpers for the local processing pipeline."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory when needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def assert_readable_file(path: Path, label: str) -> None:
    """Raise a clear error when a required file is missing or unreadable."""

    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
