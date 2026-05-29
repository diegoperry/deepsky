"""FastAPI app for DeepSky's temporary-memory upload workflow."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from deepsky_processor.web.image_processing import process_uploaded_image
from deepsky_processor.web.job_queue import enqueue_uploaded_file, get_job_status, read_job_result
from deepsky_processor.web.pipeline_worker import run_uploaded_file_pipeline
from deepsky_processor.web.storage import print_storage_doctor


STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("DEEPSKY_STORAGE_DOCTOR_ON_STARTUP", "1") == "1":
        run_smoke = os.environ.get("DEEPSKY_STORAGE_SMOKE_ON_STARTUP", "0") == "1"
        print_storage_doctor(run_smoke=run_smoke)
    yield


app = FastAPI(title="DeepSky", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "storage": "temporary-worker-job"}


@app.post("/api/process")
async def process_image(file: UploadFile = File(...)) -> Response:
    try:
        if _use_opencv_mvp():
            content = await file.read()
            processed = process_uploaded_image(content, file.filename or "upload")
        else:
            processed = await run_uploaded_file_pipeline(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        content=processed.content,
        media_type=processed.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{processed.filename}"',
            "Cache-Control": "no-store",
            "X-DeepSky-Storage": "none",
        },
    )


@app.post("/api/jobs", status_code=202)
async def create_job(file: UploadFile = File(...)) -> dict[str, object]:
    try:
        return await enqueue_uploaded_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    try:
        status = get_job_status(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {key: value for key, value in status.items() if key != "result_path"}


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> Response:
    try:
        content = read_job_result(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}_deepsky.png"',
            "Cache-Control": "no-store",
            "X-DeepSky-Storage": "temporary-job",
        },
    )


def _use_opencv_mvp() -> bool:
    import os

    return os.environ.get("DEEPSKY_WEB_PROCESSOR", "worker").lower() == "opencv"


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
