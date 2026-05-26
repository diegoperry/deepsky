"""Offline target identification using local OpenNGC-style catalog data."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PACKAGE_ROOT / "data" / "openngc_seed.csv"


@dataclass(frozen=True)
class TargetInfo:
    name: str
    object_type: str
    profile: str
    source: str


@dataclass(frozen=True)
class CatalogObject:
    name: str
    object_type: str
    ra_deg: float | None
    dec_deg: float | None
    aliases: tuple[str, ...]


def identify_target(input_path: Path) -> TargetInfo | None:
    """Identify a FITS/image target from OBJECT, filename, or RA/DEC."""

    input_path = input_path.expanduser().resolve()
    header = read_fits_header_values(input_path) if input_path.suffix.lower() in {".fit", ".fits", ".fts"} else {}
    catalog = load_openngc_catalog()
    object_name = header.get("OBJECT")
    if object_name:
        match = _match_by_name(object_name, catalog)
        if match:
            return _target_info(match, f"FITS OBJECT={object_name}")

    match = _match_by_name(input_path.stem, catalog)
    if match:
        return _target_info(match, f"filename={input_path.name}")

    ra = _float_value(header.get("CRVAL1") or header.get("RA"))
    dec = _float_value(header.get("CRVAL2") or header.get("DEC"))
    if ra is not None and dec is not None:
        match = _match_by_position(ra, dec, catalog)
        if match:
            return _target_info(match, f"FITS RA/DEC={ra:.4f},{dec:.4f}")

    return None


def load_openngc_catalog() -> list[CatalogObject]:
    """Load a local OpenNGC-style CSV.

    Set DEEPSKY_OPENNGC_CATALOG to a full local OpenNGC CSV when available. The
    bundled seed catalog covers common test objects and uses the same local
    loading path.
    """

    catalog_path = Path(os.environ.get("DEEPSKY_OPENNGC_CATALOG", DEFAULT_CATALOG))
    if not catalog_path.is_file():
        return []
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_catalog_object(row) for row in reader]


def read_fits_header_values(input_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with input_path.open("rb") as handle:
        while True:
            block = handle.read(2880)
            if not block:
                break
            for index in range(0, len(block), 80):
                card = block[index : index + 80].decode("ascii", errors="ignore")
                keyword = card[:8].strip()
                if keyword == "END":
                    return values
                if keyword and "=" in card[:10]:
                    raw_value = card[10:80].split("/", 1)[0].strip().strip("' ")
                    if raw_value:
                        values[keyword] = raw_value
    return values


def _catalog_object(row: dict[str, str]) -> CatalogObject:
    aliases = tuple(
        alias.strip()
        for alias in (row.get("aliases") or row.get("Identifiers") or row.get("Common names") or "").split(";")
        if alias.strip()
    )
    return CatalogObject(
        name=(row.get("name") or row.get("Name") or "").strip(),
        object_type=(row.get("type") or row.get("Type") or "").strip(),
        ra_deg=_float_value(row.get("ra_deg") or row.get("RA")),
        dec_deg=_float_value(row.get("dec_deg") or row.get("Dec") or row.get("DEC")),
        aliases=aliases,
    )


def _match_by_name(value: str, catalog: list[CatalogObject]) -> CatalogObject | None:
    tokens = _name_tokens(value)
    if not tokens:
        return None
    for item in catalog:
        names = (item.name, *item.aliases)
        if any(tokens == _name_tokens(name) for name in names):
            return item
    for item in catalog:
        names = (item.name, *item.aliases)
        if any(tokens in _name_tokens(name) or _name_tokens(name) in tokens for name in names):
            return item
    return None


def _match_by_position(ra: float, dec: float, catalog: list[CatalogObject]) -> CatalogObject | None:
    candidates = [item for item in catalog if item.ra_deg is not None and item.dec_deg is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda item: _angular_distance(ra, dec, item.ra_deg or 0.0, item.dec_deg or 0.0))
    return best if _angular_distance(ra, dec, best.ra_deg or 0.0, best.dec_deg or 0.0) <= 1.5 else None


def _target_info(item: CatalogObject, source: str) -> TargetInfo:
    return TargetInfo(
        name=item.name,
        object_type=item.object_type,
        profile=_profile_for_type(item.object_type),
        source=source,
    )


def _profile_for_type(object_type: str) -> str:
    normalized = object_type.strip().upper()
    if normalized.startswith("G"):
        return "galaxy"
    if normalized in {"N", "EN", "RN", "HII", "SNR"} or "NEB" in normalized:
        return "nebula"
    if normalized in {"OC", "GC", "ASTERISM"}:
        return "cluster"
    return "unknown"


def _name_tokens(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _float_value(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1_rad, dec1_rad = math.radians(ra1), math.radians(dec1)
    ra2_rad, dec2_rad = math.radians(ra2), math.radians(dec2)
    cos_angle = (
        math.sin(dec1_rad) * math.sin(dec2_rad)
        + math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
