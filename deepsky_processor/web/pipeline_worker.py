"""Temporary web-to-worker bridge for the real DeepSky pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from deepsky_processor.pipeline.job_layout import create_job_layout


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_JOBS_ROOT = PROJECT_ROOT / "deepsky_processor" / "jobs" / "web-requests"
DEFAULT_STARNET_CONTAINER_PATH = "/app/tools/StarNet/linux/StarNetv2CLI_linux/starnet++"
DEFAULT_SCUNET_CONTAINER_PATH = "/app/models/SCUNet/scunet_color_real_gan.pth"


@dataclass(frozen=True)
class WebPipelineResult:
    filename: str
    media_type: str
    content: bytes


async def run_uploaded_file_pipeline(upload: UploadFile) -> WebPipelineResult:
    """Run one uploaded frame through the real Docker worker pipeline."""

    original_name = upload.filename or "upload.tif"
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".fit", ".fits", ".fts", ".tif", ".tiff"}:
        raise ValueError(
            "The worker web pipeline currently accepts single FITS or 16-bit TIFF files. "
            "Multi-frame lights/darks/flats/biases jobs are available from the CLI pipeline."
        )

    job_id = f"web-{uuid.uuid4().hex}"
    layout = create_job_layout(WEB_JOBS_ROOT / job_id)
    try:
        safe_name = _safe_filename(original_name)
        uploaded_path = layout.root / safe_name
        content = await upload.read()
        if not content:
            raise ValueError("Uploaded file is empty")
        uploaded_path.write_bytes(content)

        _run_worker_container(uploaded_path, layout.output)

        final_image = layout.output / "final.png"
        if not final_image.is_file():
            raise RuntimeError(f"Worker completed but final image was not created: {final_image}")

        return WebPipelineResult(
            filename=f"{Path(safe_name).stem}_deepsky.png",
            media_type="image/png",
            content=final_image.read_bytes(),
        )
    finally:
        shutil.rmtree(layout.root, ignore_errors=True)


def _run_worker_container(input_path: Path, output_dir: Path) -> None:
    image = os.environ.get("DEEPSKY_WORKER_IMAGE", "deepsky-worker")
    timeout = int(os.environ.get("DEEPSKY_WORKER_TIMEOUT_SECONDS", "1800"))
    container_input_path = _container_path(input_path)
    container_output_dir = _container_path(output_dir)

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{PROJECT_ROOT}:/app",
        "-e",
        f"STARNET_PATH={os.environ.get('STARNET_PATH', DEFAULT_STARNET_CONTAINER_PATH)}",
        "-e",
        f"SCUNET_MODEL_PATH={os.environ.get('SCUNET_MODEL_PATH', DEFAULT_SCUNET_CONTAINER_PATH)}",
        "-e",
        f"SCUNET_MODEL_TYPE={os.environ.get('SCUNET_MODEL_TYPE', 'official')}",
        "-e",
        f"SCUNET_DEVICE={os.environ.get('SCUNET_DEVICE', 'cpu')}",
        image,
        "python",
        "-m",
        "deepsky_processor.pipeline.main_pipeline",
        "--run-single-image",
        "--mode",
        "container",
        "--input",
        container_input_path,
        "--workdir",
        container_output_dir,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RuntimeError("Docker was not found. Install Docker Desktop and start it.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"DeepSky worker timed out after {timeout} seconds") from exc
    except subprocess.CalledProcessError as exc:
        output = "\n".join(part for part in (exc.stdout, exc.stderr) if part).strip()
        raise RuntimeError(f"DeepSky worker failed:\n{output or exc}") from exc


def _container_path(path: Path) -> str:
    relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    return "/app/" + relative.as_posix()


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip().replace(" ", "_")
    allowed = []
    for char in name:
        if char.isalnum() or char in {".", "-", "_"}:
            allowed.append(char)
    cleaned = "".join(allowed).strip("._")
    return cleaned or "upload.tif"
