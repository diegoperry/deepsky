"""Single-image real-tool processing for web uploads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess

import cv2
import numpy as np

from deepsky_processor.pipeline.config import PipelineConfig
from deepsky_processor.pipeline.opencv_runner import (
    compose_deepsky_image,
    prepare_starnet_input_file,
)
from deepsky_processor.pipeline.scunet_runner import run_scunet
from deepsky_processor.pipeline.siril_runner import run_siril_script
from deepsky_processor.pipeline.starnet_runner import run_starnet
from deepsky_processor.pipeline.target_catalog import identify_target, read_fits_header_values


@dataclass(frozen=True)
class SingleImageArtifacts:
    siril_prepared: Path
    starnet_input: Path
    starless: Path
    denoised: Path
    final_image: Path


def run_single_image_pipeline(
    input_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    mode: str,
) -> SingleImageArtifacts:
    """Run Siril -> StarNet++ -> SCUNet -> OpenCV on one FITS or 16-bit TIFF image."""

    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_starnet_input(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = identify_target(input_path)
    _write_target_identification_log(output_dir, target)

    siril_command = "siril-cli" if mode == "container" else config.siril_cli
    siril_prepared = output_dir / "siril_prepared.tif"
    siril_color_calibrated = _run_siril_single_image_prepare(
        input_path=input_path,
        output_path=siril_prepared,
        output_dir=output_dir,
        siril_command=siril_command,
    )
    if not siril_prepared.is_file():
        raise RuntimeError(f"Siril completed but did not create {siril_prepared}")

    starnet_input = output_dir / "starnet_input.tif"
    prepare_starnet_input_file(siril_prepared, starnet_input)

    starless = output_dir / "starless.tif"
    run_starnet(
        input_path=starnet_input,
        output_path=starless,
        starnet_path=config.starnet_path,
        args_template=config.starnet_args,
        require_executable=mode == "container",
    )

    denoised = output_dir / "denoised.tif"
    run_scunet(
        input_path=starless,
        output_path=denoised,
        model_path=config.scunet_model_path,
        device=config.scunet_device,
        model_type=config.scunet_model_type,
    )

    final_image = output_dir / "final.png"
    compose_deepsky_image(
        original_path=starnet_input,
        starless_path=starless,
        denoised_path=denoised,
        output_path=final_image,
        preserve_color_calibration=siril_color_calibrated,
        target_profile=target.profile if target else None,
        star_source_path=siril_prepared,
    )

    return SingleImageArtifacts(
        siril_prepared=siril_prepared,
        starnet_input=starnet_input,
        starless=starless,
        denoised=denoised,
        final_image=final_image,
    )


def _validate_starnet_input(input_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Single-image input does not exist: {input_path}")
    if input_path.suffix.lower() in {".fit", ".fits", ".fts"}:
        return
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(
            "Single-image web processing currently requires FITS or 16-bit TIFF input."
        )

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read input TIFF: {input_path}")
    if image.dtype != np.uint16:
        raise ValueError(
            "Single-image web processing currently requires a 16-bit TIFF. "
            f"Received {image.dtype}."
        )
    if image.ndim == 3 and image.shape[2] not in {1, 3, 4}:
        raise ValueError(f"Unsupported TIFF channel count for StarNet++: {image.shape}")


def _run_siril_single_image_prepare(
    input_path: Path,
    output_path: Path,
    output_dir: Path,
    siril_command: str,
) -> bool:
    pcc_command = _build_siril_pcc_command(input_path)
    script_path = output_dir / "siril_single_image_prepare.ssf"
    script_path.write_text(
        _build_siril_prepare_script(input_path, output_path, pcc_command),
        encoding="utf-8",
    )
    try:
        run_siril_script(
            script_path=script_path,
            working_directory=output_dir,
            command=siril_command,
            env=os.environ,
        )
        _write_siril_color_log(output_dir, "Siril Photometric Color Calibration: applied")
        return pcc_command is not None
    except subprocess.CalledProcessError as exc:
        if pcc_command is None or not _is_siril_catalog_unavailable_error(exc):
            raise
        _write_siril_color_log(
            output_dir,
            "\n".join(
                [
                    "Siril Photometric Color Calibration: unavailable",
                    "Reason: this Siril build could not access/download the required star catalog.",
                    "The pipeline reran Siril preparation without PCC instead of faking color calibration.",
                    "",
                    "Siril output:",
                    (exc.stdout or exc.stderr or str(exc)).strip(),
                ]
            ),
        )
        script_path.write_text(
            _build_siril_prepare_script(input_path, output_path, None, allow_pcc=False),
            encoding="utf-8",
        )
        run_siril_script(
            script_path=script_path,
            working_directory=output_dir,
            command=siril_command,
            env=os.environ,
        )
        return False


def _build_siril_prepare_script(
    input_path: Path,
    output_path: Path,
    pcc_command: str | None = None,
    allow_pcc: bool = True,
) -> str:
    if pcc_command is None and allow_pcc:
        pcc_command = _build_siril_pcc_command(input_path)
    pcc_lines = (
        [
            "# Photometric Color Calibration uses real Siril plate solving/catalog data.",
            pcc_command,
        ]
        if pcc_command
        else [
            "# Photometric Color Calibration skipped: FITS center/focal/pixel metadata not found.",
        ]
    )
    return "\n".join(
        [
            "# DeepSky single-image Siril preparation script",
            "requires 1.2.0",
            f'cd "{input_path.parent.as_posix()}"',
            f'load "{input_path.name}"',
            *pcc_lines,
            f'savetif "{output_path.with_suffix("").as_posix()}"',
            "close",
            "",
        ]
    )


def _has_siril_pcc_metadata(input_path: Path) -> bool:
    return _build_siril_pcc_command(input_path) is not None


def _is_siril_catalog_unavailable_error(exc: subprocess.CalledProcessError) -> bool:
    output = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
    known_messages = [
        "compiled without networking support",
        "Could not download the online star catalogue",
        "No catalog",
    ]
    return any(message in output for message in known_messages)


def _write_siril_color_log(output_dir: Path, message: str) -> None:
    (output_dir / "siril_color_calibration.log").write_text(message + "\n", encoding="utf-8")


def _write_target_identification_log(output_dir: Path, target: object | None) -> None:
    if target is None:
        message = "OpenNGC target identification: no local match"
    else:
        message = (
            "OpenNGC target identification: "
            f"{target.name} ({target.object_type}) -> profile={target.profile}; source={target.source}"
        )
    (output_dir / "target_identification.log").write_text(message + "\n", encoding="utf-8")


def _build_siril_pcc_command(input_path: Path) -> str | None:
    if input_path.suffix.lower() not in {".fit", ".fits", ".fts"}:
        return None
    metadata = _read_fits_header_values(input_path)
    ra = metadata.get("CRVAL1") or metadata.get("RA")
    dec = metadata.get("CRVAL2") or metadata.get("DEC")
    focal = metadata.get("FOCALLEN") or metadata.get("FOCLEN")
    pixel_size = metadata.get("XPIXSZ") or metadata.get("PIXSIZE1") or metadata.get("YPIXSZ")
    if None in {ra, dec, focal, pixel_size}:
        return None
    return (
        f"pcc {ra},{dec} -noflip -platesolve "
        f"-focal={focal} -pixelsize={pixel_size} -downscale -catalog=apass"
    )


def _read_fits_header_values(input_path: Path) -> dict[str, str]:
    return {
        key: value
        for key, value in read_fits_header_values(input_path).items()
        if _looks_numeric(value)
    }


def _looks_numeric(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", value))
