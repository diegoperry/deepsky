"""Full real-tool DeepSky pipeline orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from deepsky_processor.pipeline.config import PipelineConfig
from deepsky_processor.pipeline.job_layout import (
    JobLayout,
    validate_job_layout,
    validate_profile_inputs,
)
from deepsky_processor.pipeline.opencv_runner import stretch_image_file
from deepsky_processor.pipeline.scunet_runner import run_scunet
from deepsky_processor.pipeline.siril_script_builder import write_preprocess_script
from deepsky_processor.pipeline.siril_runner import run_siril_script
from deepsky_processor.pipeline.starnet_runner import run_starnet


STACK_CANDIDATES = (
    "result.fit",
    "result.fits",
    "result.fts",
    "result.tif",
    "result.tiff",
    "result.png",
)


@dataclass(frozen=True)
class PipelineArtifacts:
    siril_stack: Path
    starless: Path
    denoised: Path
    final_image: Path


def run_full_pipeline(
    job_dir: Path,
    config: PipelineConfig,
    mode: str,
    siril_profile: str,
) -> PipelineArtifacts:
    """Run the full Siril -> StarNet++ -> SCUNet -> OpenCV pipeline."""

    layout = validate_job_layout(job_dir)
    validate_profile_inputs(layout, siril_profile)
    script_path = write_preprocess_script(layout, siril_profile)

    siril_command = "siril-cli" if mode == "container" else config.siril_cli
    siril_result = run_siril_script(
        script_path=script_path,
        working_directory=layout.root,
        command=siril_command,
        env=os.environ,
    )
    _write_stage_log(layout, "01_siril", siril_result.stdout, siril_result.stderr)

    siril_stack = _find_siril_stack(layout)
    starless = layout.output / "starless.tif"
    starnet_result = run_starnet(
        input_path=siril_stack,
        output_path=starless,
        starnet_path=config.starnet_path,
        args_template=config.starnet_args,
        require_executable=mode == "container",
    )
    _write_stage_log(layout, "02_starnet", starnet_result.stdout, starnet_result.stderr)

    denoised = layout.output / "denoised.tif"
    run_scunet(
        input_path=starless,
        output_path=denoised,
        model_path=config.scunet_model_path,
        device=config.scunet_device,
        model_type=config.scunet_model_type,
    )
    _write_stage_log(layout, "03_scunet", f"SCUNet output: {denoised}\n", "")

    final_image = layout.output / "final.png"
    stretch_image_file(denoised, final_image)
    _write_stage_log(layout, "04_opencv", f"Final image: {final_image}\n", "")

    return PipelineArtifacts(
        siril_stack=siril_stack,
        starless=starless,
        denoised=denoised,
        final_image=final_image,
    )


def _find_siril_stack(layout: JobLayout) -> Path:
    for name in STACK_CANDIDATES:
        candidate = layout.output / name
        if candidate.exists() and candidate.is_file():
            return candidate
    found = sorted(path.name for path in layout.output.iterdir() if path.is_file())
    raise FileNotFoundError(
        "Siril completed, but no expected stacked output was found. "
        f"Looked for {', '.join(STACK_CANDIDATES)} in {layout.output}. "
        f"Found: {', '.join(found) if found else 'no files'}"
    )


def _write_stage_log(layout: JobLayout, stage: str, stdout: str, stderr: str) -> None:
    layout.logs.mkdir(parents=True, exist_ok=True)
    (layout.logs / f"{stage}.stdout.log").write_text(stdout or "", encoding="utf-8")
    (layout.logs / f"{stage}.stderr.log").write_text(stderr or "", encoding="utf-8")
