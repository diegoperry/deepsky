from __future__ import annotations

import sys
import types
from io import BytesIO

import pytest

import deepsky_processor.web.storage as storage


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_boto(monkeypatch, calls):
    fake_boto3 = types.ModuleType("boto3")

    def client(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return object()

    fake_boto3.client = client

    fake_botocore = types.ModuleType("botocore")
    fake_config = types.ModuleType("botocore.config")
    fake_config.Config = FakeConfig

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config)


def test_r2_client_uses_cloudflare_path_style_config(monkeypatch):
    calls = []
    _install_fake_boto(monkeypatch, calls)
    monkeypatch.setenv("R2_ENDPOINT_URL", " https://account.r2.cloudflarestorage.com ")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", " access-key ")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", " secret-key ")
    monkeypatch.setenv("R2_BUCKET", "deepsky")

    storage._s3()

    assert calls
    kwargs = calls[0]["kwargs"]
    assert calls[0]["args"] == ("s3",)
    assert kwargs["endpoint_url"] == "https://account.r2.cloudflarestorage.com"
    assert kwargs["aws_access_key_id"] == "access-key"
    assert kwargs["aws_secret_access_key"] == "secret-key"
    assert kwargs["region_name"] == "auto"
    assert kwargs["config"].kwargs == {
        "signature_version": "s3v4",
        "s3": {"addressing_style": "path"},
    }


def test_r2_endpoint_rejects_bucket_path(monkeypatch):
    calls = []
    _install_fake_boto(monkeypatch, calls)
    monkeypatch.setenv(
        "R2_ENDPOINT_URL",
        "https://account.r2.cloudflarestorage.com/deepsky",
    )
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret-key")
    monkeypatch.setenv("R2_BUCKET", "deepsky")

    with pytest.raises(RuntimeError, match="must not include a bucket or path"):
        storage._s3()


def test_r2_upload_passes_bucket_separately(monkeypatch):
    captured = {}

    class FakeClient:
        def put_object(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DEEPSKY_STORAGE_BACKEND", "r2")
    monkeypatch.setenv("R2_BUCKET", "deepsky")
    monkeypatch.setattr(storage, "_s3", lambda: FakeClient())

    storage.upload_bytes("railway/web-123/input/galaxy.tif", b"image")

    assert captured["Bucket"] == "deepsky"
    assert captured["Key"] == "railway/web-123/input/galaxy.tif"
    assert captured["Body"] == b"image"


def test_r2_worker_backend_uses_private_worker_endpoint(monkeypatch):
    requests = []

    class FakeResponse(BytesIO):
        pass

    def fake_http_request(request, timeout):
        requests.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": request.data,
                "timeout": timeout,
            }
        )
        return FakeResponse(b"stored")

    monkeypatch.setenv("DEEPSKY_STORAGE_BACKEND", "r2_worker")
    monkeypatch.setenv("R2_WORKER_URL", " https://deepsky-r2.example.workers.dev/ ")
    monkeypatch.setenv("R2_WORKER_TOKEN", " secret-token ")
    monkeypatch.setattr(storage, "_http_request", fake_http_request)

    storage.upload_bytes("railway/web-123/input/galaxy.tif", b"image", "image/tiff")
    content = storage.read_bytes("railway/web-123/input/galaxy.tif")

    assert content == b"stored"
    assert requests[0]["method"] == "PUT"
    assert requests[0]["url"] == "https://deepsky-r2.example.workers.dev/objects/railway/web-123/input/galaxy.tif"
    assert requests[0]["headers"]["Authorization"] == "Bearer secret-token"
    assert requests[0]["headers"]["Content-type"] == "image/tiff"
    assert requests[0]["body"] == b"image"
    assert requests[1]["method"] == "GET"
