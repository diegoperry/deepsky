"""Redis/RQ job helpers for temporary DeepSky web processing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from deepsky_processor.web import storage
from deepsky_processor.web.pipeline_worker import WEB_JOBS_ROOT, _safe_filename


QUEUE_NAME = os.environ.get("DEEPSKY_QUEUE_NAME", "deepsky")
DEFAULT_REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
JOB_MAX_AGE_SECONDS = int(os.environ.get("DEEPSKY_WEB_JOB_MAX_AGE_SECONDS", "3600"))
WORKER_TIMEOUT_SECONDS = int(os.environ.get("DEEPSKY_WORKER_TIMEOUT_SECONDS", "1800"))


async def enqueue_uploaded_file(upload: UploadFile) -> dict[str, Any]:
    """Save an upload in shared storage and enqueue it for a real worker."""

    _cleanup_expired_jobs()
    original_name = upload.filename or "upload.tif"
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".fit", ".fits", ".fts", ".tif", ".tiff"}:
        raise ValueError("DeepSky accepts FITS, FIT, FTS, TIF, and TIFF uploads.")

    job_id = f"web-{uuid.uuid4().hex}"
    safe_name = _safe_filename(original_name)
    content = await upload.read()
    if not content:
        raise ValueError("Uploaded file is empty")

    input_key = storage.job_key(job_id, "input", safe_name)
    status_key = storage.job_key(job_id, "status.json")
    metadata_key = storage.job_key(job_id, "metadata.json")
    result_key = storage.job_key(job_id, "output", "final.png")
    content_type = _content_type_for(safe_name)
    now = time.time()
    storage.upload_bytes(input_key, content, content_type=content_type)
    storage.write_json(
        metadata_key,
        {
            "job_id": job_id,
            "filename": safe_name,
            "content_type": content_type,
            "size_bytes": len(content),
            "input_key": input_key,
            "status_key": status_key,
            "result_key": result_key,
            "created_at": now,
        },
    )
    _write_status_key(
        status_key,
        _initial_status(
            job_id,
            safe_name,
            input_key,
            status_key,
            metadata_key,
            result_key,
            now=now,
        ),
    )

    queue = _get_queue()
    queue.enqueue(
        process_queued_job,
        job_id,
        {
            "input_key": input_key,
            "status_key": status_key,
            "metadata_key": metadata_key,
            "result_key": result_key,
            "filename": safe_name,
        },
        job_timeout=WORKER_TIMEOUT_SECONDS + 60,
        result_ttl=JOB_MAX_AGE_SECONDS,
        failure_ttl=JOB_MAX_AGE_SECONDS,
    )
    return get_job_status(job_id)


def process_queued_job(job_id: str, payload: dict[str, str] | str, output_dir: str | None = None) -> None:
    """Run one queued web job inside the Linux worker environment."""

    if isinstance(payload, dict):
        _process_storage_backed_job(job_id, payload)
        return

    # Legacy local-path payload support for old in-flight local development jobs.
    if output_dir is None:
        raise RuntimeError("Queued job payload is missing output_dir")
    input_path = payload
    input_file = Path(input_path)
    output_path = Path(output_dir)
    job_root = input_file.parents[1]
    try:
        _patch_status(job_root, status="running", progress=8, step="Starting worker")
        _run_pipeline_process(input_file, output_path, job_root)
        final_image = output_path / "final.png"
        if not final_image.is_file():
            raise RuntimeError(f"Worker completed but final image was not created: {final_image}")
        _patch_status(
            job_root,
            status="finished",
            progress=100,
            step="Processing complete",
            result_url=f"/api/jobs/{job_id}/result",
            result_path=str(final_image),
        )
    except Exception as exc:  # noqa: BLE001 - job status should preserve worker failures.
        _patch_status(
            job_root,
            status="failed",
            progress=_read_status(job_root).get("progress", 0),
            step="Processing failed",
            error=str(exc),
        )
        raise


def _process_storage_backed_job(job_id: str, payload: dict[str, str]) -> None:
    status_key = payload["status_key"]
    result_key = payload["result_key"]
    filename = payload["filename"]
    try:
        _patch_status_key(status_key, status="running", progress=8, step="Starting worker")
        with tempfile.TemporaryDirectory(prefix=f"{job_id}-") as temp_dir:
            temp_root = Path(temp_dir)
            input_file = temp_root / "input" / filename
            output_dir = temp_root / "output"
            storage.download_file(status_key, temp_root / "status.json")
            if payload.get("metadata_key"):
                storage.download_file(payload["metadata_key"], temp_root / "metadata.json")
            storage.download_file(payload["input_key"], input_file)
            output_dir.mkdir(parents=True, exist_ok=True)

            _run_pipeline_process_for_status_key(input_file, output_dir, status_key)

            final_image = output_dir / "final.png"
            if not final_image.is_file():
                raise RuntimeError(f"Worker completed but final image was not created: {final_image}")
            storage.upload_file(result_key, final_image, content_type="image/png")
        _patch_status_key(
            status_key,
            status="finished",
            progress=100,
            step="Processing complete",
            result_url=f"/api/jobs/{job_id}/result",
            result_key=result_key,
            result_path=None,
        )
    except Exception as exc:  # noqa: BLE001 - job status should preserve worker failures.
        try:
            current = _read_status_key(status_key)
            progress = current.get("progress", 0)
        except Exception:  # noqa: BLE001 - if status is gone, still report failure to RQ.
            progress = 0
        _patch_status_key(
            status_key,
            status="failed",
            progress=progress,
            step="Processing failed",
            error=str(exc),
        )
        raise


def get_job_status(job_id: str) -> dict[str, Any]:
    _cleanup_expired_jobs()
    status_key = storage.job_key(job_id, "status.json")
    if not storage.exists(status_key):
        raise FileNotFoundError(f"Job was not found or has expired: {job_id}")
    return _read_status_key(status_key)


def read_job_result(job_id: str) -> bytes:
    status = get_job_status(job_id)
    if status.get("status") != "finished":
        raise RuntimeError(f"Job is not finished yet: {job_id}")
    result_key = status.get("result_key")
    if result_key:
        if not storage.exists(result_key):
            raise FileNotFoundError(f"Result image was not found for job: {job_id}")
        return storage.read_bytes(result_key)

    result_path = Path(status.get("result_path") or _job_root(job_id) / "output" / "final.png")
    if not result_path.is_file():
        raise FileNotFoundError(f"Result image was not found for job: {job_id}")
    return result_path.read_bytes()


def _run_pipeline_process(input_path: Path, output_dir: Path, job_root: Path) -> None:
    _patch_status(job_root, progress=16, step="Siril FITS preparation")
    _run_pipeline_process_impl(input_path, output_dir)


def _run_pipeline_process_for_status_key(input_path: Path, output_dir: Path, status_key: str) -> None:
    _patch_status_key(status_key, progress=16, step="Siril FITS preparation")
    _run_pipeline_process_impl(input_path, output_dir)


def _run_pipeline_process_impl(input_path: Path, output_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "deepsky_processor.pipeline.main_pipeline",
        "--run-single-image",
        "--mode",
        os.environ.get("DEEPSKY_VERIFY_MODE", "container"),
        "--input",
        str(input_path),
        "--workdir",
        str(output_dir),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(output or f"DeepSky worker exited with code {result.returncode}")


def _get_queue():
    try:
        from redis import Redis
        from rq import Queue
    except ImportError as exc:
        raise RuntimeError(
            "Redis queue dependencies are not installed. Run: python -m pip install redis rq"
        ) from exc

    connection = Redis.from_url(os.environ.get("REDIS_URL", DEFAULT_REDIS_URL))
    return Queue(QUEUE_NAME, connection=connection)


def _job_root(job_id: str) -> Path:
    if not job_id.startswith("web-") or any(char in job_id for char in "/\\"):
        raise FileNotFoundError(f"Invalid job id: {job_id}")
    return WEB_JOBS_ROOT / job_id


def _initial_status(
    job_id: str,
    filename: str,
    input_key: str,
    status_key: str,
    metadata_key: str,
    result_key: str,
    now: float | None = None,
) -> dict[str, Any]:
    now = now or time.time()
    return {
        "job_id": job_id,
        "status": "queued",
        "progress": 3,
        "step": "Upload received",
        "filename": filename,
        "storage_backend": storage.backend_name(),
        "input_key": input_key,
        "status_key": status_key,
        "metadata_key": metadata_key,
        "result_key": result_key,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + JOB_MAX_AGE_SECONDS,
        "result_url": None,
        "result_path": None,
        "error": None,
    }


def _read_status(job_root: Path) -> dict[str, Any]:
    return json.loads((job_root / "status.json").read_text(encoding="utf-8"))


def _write_status(job_root: Path, status: dict[str, Any]) -> None:
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _patch_status(job_root: Path, **changes: Any) -> None:
    status = _read_status(job_root)
    status.update(changes)
    status["updated_at"] = time.time()
    _write_status(job_root, status)


def _read_status_key(status_key: str) -> dict[str, Any]:
    return storage.read_json(status_key)


def _write_status_key(status_key: str, status: dict[str, Any]) -> None:
    storage.write_json(status_key, status)


def _patch_status_key(status_key: str, **changes: Any) -> None:
    status = _read_status_key(status_key)
    status.update(changes)
    status["updated_at"] = time.time()
    _write_status_key(status_key, status)


def _content_type_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".fit", ".fits", ".fts"}:
        return "application/fits"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    return "application/octet-stream"


def _cleanup_expired_jobs() -> None:
    if storage.backend_name() != "local":
        return
    root = WEB_JOBS_ROOT
    if not root.is_dir():
        return
    now = time.time()
    for job_root in root.glob("web-*"):
        if not job_root.is_dir():
            continue
        status_path = job_root / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            expired = float(status.get("expires_at", 0)) < now
        except Exception:  # noqa: BLE001 - stale partial job dirs are temporary.
            expired = (now - job_root.stat().st_mtime) > JOB_MAX_AGE_SECONDS
        if expired:
            shutil.rmtree(job_root, ignore_errors=True)
