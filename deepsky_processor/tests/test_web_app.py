from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

import deepsky_processor.web.app as web_app
import deepsky_processor.web.job_queue as job_queue
import deepsky_processor.web.pipeline_worker as pipeline_worker
import deepsky_processor.web.storage as storage
from deepsky_processor.web.app import app
from deepsky_processor.web.image_processing import process_uploaded_image
from deepsky_processor.web.pipeline_worker import WebPipelineResult


def _sample_png_bytes() -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint16)
    image[:, :, 0] = np.linspace(0, 65535, 32, dtype=np.uint16)
    image[:, :, 1] = 8000
    image[10:20, 10:20, 2] = 50000
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_process_uploaded_image_returns_png_bytes() -> None:
    result = process_uploaded_image(_sample_png_bytes(), "nebula.png")

    assert result.filename == "nebula_deepsky.png"
    assert result.media_type == "image/png"
    assert result.content.startswith(b"\x89PNG")


def test_process_endpoint_returns_no_store_png(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSKY_WEB_PROCESSOR", "opencv")
    client = TestClient(app)

    response = client.post(
        "/api/process",
        files={"file": ("nebula.png", _sample_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-deepsky-storage"] == "none"
    assert response.content.startswith(b"\x89PNG")


def test_process_endpoint_can_use_worker_pipeline(monkeypatch) -> None:
    async def fake_worker(file):
        assert file.filename == "nebula.tif"
        return WebPipelineResult(
            filename="nebula_deepsky.png",
            media_type="image/png",
            content=b"\x89PNG worker",
        )

    monkeypatch.delenv("DEEPSKY_WEB_PROCESSOR", raising=False)
    monkeypatch.setattr(web_app, "run_uploaded_file_pipeline", fake_worker)
    client = TestClient(app)

    response = client.post(
        "/api/process",
        files={"file": ("nebula.tif", _sample_png_bytes(), "image/tiff")},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-deepsky-storage"] == "none"
    assert response.content == b"\x89PNG worker"


def test_process_endpoint_accepts_fits_for_worker_pipeline(monkeypatch) -> None:
    async def fake_worker(file):
        assert file.filename == "heart.fits"
        return WebPipelineResult(
            filename="heart_deepsky.png",
            media_type="image/png",
            content=b"\x89PNG fits worker",
        )

    monkeypatch.delenv("DEEPSKY_WEB_PROCESSOR", raising=False)
    monkeypatch.setattr(web_app, "run_uploaded_file_pipeline", fake_worker)
    client = TestClient(app)

    response = client.post(
        "/api/process",
        files={"file": ("heart.fits", b"SIMPLE  = T", "image/fits")},
    )

    assert response.status_code == 200
    assert response.content == b"\x89PNG fits worker"


def test_create_job_endpoint_returns_queue_status(monkeypatch) -> None:
    async def fake_enqueue(file):
        assert file.filename == "galaxy.fit"
        return {
            "job_id": "web-123",
            "status": "queued",
            "progress": 3,
            "step": "Upload received",
            "result_url": None,
        }

    monkeypatch.setattr(web_app, "enqueue_uploaded_file", fake_enqueue)
    client = TestClient(app)

    response = client.post(
        "/api/jobs",
        files={"file": ("galaxy.fit", b"SIMPLE  = T", "image/fits")},
    )

    assert response.status_code == 202
    assert response.json()["job_id"] == "web-123"
    assert response.json()["status"] == "queued"


def test_job_status_endpoint_hides_result_path(monkeypatch) -> None:
    def fake_status(job_id):
        assert job_id == "web-123"
        return {
            "job_id": job_id,
            "status": "finished",
            "progress": 100,
            "step": "Processing complete",
            "result_url": "/api/jobs/web-123/result",
            "result_path": "/tmp/private/final.png",
        }

    monkeypatch.setattr(web_app, "get_job_status", fake_status)
    client = TestClient(app)

    response = client.get("/api/jobs/web-123")

    assert response.status_code == 200
    assert response.json()["status"] == "finished"
    assert "result_path" not in response.json()


def test_job_result_endpoint_returns_no_store_png(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "read_job_result", lambda job_id: b"\x89PNG queued")
    client = TestClient(app)

    response = client.get("/api/jobs/web-123/result")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-deepsky-storage"] == "temporary-job"
    assert response.content == b"\x89PNG queued"


def test_process_endpoint_rejects_unknown_file(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSKY_WEB_PROCESSOR", "opencv")
    client = TestClient(app)

    response = client.post(
        "/api/process",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_enqueue_uploaded_file_writes_temp_job_and_enqueues(monkeypatch, tmp_path) -> None:
    class FakeUpload:
        filename = "galaxy.fit"

        async def read(self):
            return b"SIMPLE  = T"

    class FakeQueue:
        def enqueue(self, func, *args, **kwargs):
            calls["func"] = func
            calls["args"] = args
            calls["kwargs"] = kwargs

    calls = {}
    monkeypatch.setattr(job_queue, "WEB_JOBS_ROOT", tmp_path)
    monkeypatch.setattr(storage, "LOCAL_STORAGE_ROOT", tmp_path)
    monkeypatch.delenv("DEEPSKY_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("R2_BUCKET", raising=False)
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)
    monkeypatch.setattr(job_queue, "_get_queue", lambda: FakeQueue())

    import anyio

    status = anyio.run(job_queue.enqueue_uploaded_file, FakeUpload())

    assert status["status"] == "queued"
    assert status["job_id"].startswith("web-")
    assert calls["func"] is job_queue.process_queued_job
    payload = calls["args"][1]
    assert payload["input_key"].endswith("/input/galaxy.fit")
    assert payload["status_key"].endswith("/status.json")
    assert payload["metadata_key"].endswith("/metadata.json")
    assert (tmp_path / status["job_id"] / "input" / "galaxy.fit").is_file()
    assert (tmp_path / status["job_id"] / "status.json").is_file()
    assert (tmp_path / status["job_id"] / "metadata.json").is_file()


def test_storage_backed_worker_uses_temp_paths_and_uploads_result(monkeypatch, tmp_path) -> None:
    job_id = "web-123"
    status_key = storage.job_key(job_id, "status.json")
    input_key = storage.job_key(job_id, "input", "galaxy.fit")
    result_key = storage.job_key(job_id, "output", "final.png")
    metadata_key = storage.job_key(job_id, "metadata.json")

    monkeypatch.setattr(storage, "LOCAL_STORAGE_ROOT", tmp_path)
    monkeypatch.delenv("DEEPSKY_STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("R2_BUCKET", raising=False)
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)
    storage.upload_bytes(input_key, b"fits")
    storage.write_json(metadata_key, {"job_id": job_id, "filename": "galaxy.fit"})
    storage.write_json(
        status_key,
        {
            "job_id": job_id,
            "status": "queued",
            "progress": 3,
            "step": "Upload received",
            "result_key": result_key,
            "error": None,
        },
    )

    def fake_run_pipeline(input_path, output_dir, status_path):
        assert input_path.name == "galaxy.fit"
        assert input_path.read_bytes() == b"fits"
        assert status_path == status_key
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final.png").write_bytes(b"\x89PNG final")

    monkeypatch.setattr(job_queue, "_run_pipeline_process_for_status_key", fake_run_pipeline)

    job_queue.process_queued_job(
        job_id,
        {
            "input_key": input_key,
            "status_key": status_key,
            "metadata_key": metadata_key,
            "result_key": result_key,
            "filename": "galaxy.fit",
        },
    )

    status = storage.read_json(status_key)
    assert status["status"] == "finished"
    assert status["progress"] == 100
    assert job_queue.read_job_result(job_id) == b"\x89PNG final"


def test_worker_command_uses_container_paths(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "DeepSky"
    job_dir = project_root / "deepsky_processor" / "jobs" / "web-requests" / "job"
    input_path = job_dir / "input.tif"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True)
    input_path.write_bytes(b"tiff")
    calls = {}

    def fake_run(command, check, capture_output, text, timeout):
        calls["command"] = command
        calls["check"] = check
        calls["capture_output"] = capture_output
        calls["text"] = text
        calls["timeout"] = timeout

    monkeypatch.setattr(pipeline_worker, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("STARNET_PATH", "/app/tools/StarNet/starnet++")
    monkeypatch.setenv("SCUNET_MODEL_PATH", "/app/models/SCUNet/model.pth")
    monkeypatch.setenv("SCUNET_MODEL_TYPE", "official")
    monkeypatch.setattr(pipeline_worker.subprocess, "run", fake_run)

    pipeline_worker._run_worker_container(input_path, output_dir)

    command = calls["command"]
    assert command[:5] == ["docker", "run", "--rm", "-v", f"{project_root}:/app"]
    assert "STARNET_PATH=/app/tools/StarNet/starnet++" in command
    assert "SCUNET_MODEL_PATH=/app/models/SCUNet/model.pth" in command
    assert "--run-single-image" in command
    assert "/app/deepsky_processor/jobs/web-requests/job/input.tif" in command
    assert "/app/deepsky_processor/jobs/web-requests/job/output" in command
    assert calls["check"] is True
    assert calls["capture_output"] is True


def test_worker_command_reports_docker_missing(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "DeepSky"
    job_dir = project_root / "deepsky_processor" / "jobs" / "web-requests" / "job"
    input_path = job_dir / "input.tif"
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True)
    input_path.write_bytes(b"tiff")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(pipeline_worker, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(pipeline_worker.subprocess, "run", fake_run)

    try:
        pipeline_worker._run_worker_container(input_path, output_dir)
    except RuntimeError as exc:
        assert "Docker was not found" in str(exc)
    else:
        raise AssertionError("Expected missing Docker to fail clearly")
