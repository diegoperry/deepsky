"""Shared web job storage for local development and Cloudflare R2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_STORAGE_ROOT = Path(
    os.environ.get(
        "DEEPSKY_LOCAL_STORAGE_ROOT",
        str(PROJECT_ROOT / "deepsky_processor" / "jobs" / "web-requests"),
    )
)


def backend_name() -> str:
    configured = os.environ.get("DEEPSKY_STORAGE_BACKEND")
    if configured:
        return configured.lower()
    if os.environ.get("R2_BUCKET") and os.environ.get("R2_ENDPOINT_URL"):
        return "r2"
    return "local"


def object_prefix() -> str:
    return os.environ.get("DEEPSKY_STORAGE_PREFIX", "").strip("/")


def job_key(job_id: str, *parts: str) -> str:
    if not job_id.startswith("web-") or any(char in job_id for char in "/\\"):
        raise FileNotFoundError(f"Invalid job id: {job_id}")
    segments = [segment.strip("/") for segment in (object_prefix(), job_id, *parts) if segment]
    return "/".join(segments)


def upload_bytes(key: str, content: bytes, content_type: str | None = None) -> None:
    if backend_name() == "r2":
        extra_args = {"ContentType": content_type} if content_type else None
        kwargs = {"Bucket": _bucket(), "Key": key, "Body": content}
        if extra_args:
            kwargs["ContentType"] = extra_args["ContentType"]
        _s3().put_object(**kwargs)
        return

    path = _local_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def upload_file(key: str, source_path: Path, content_type: str | None = None) -> None:
    if backend_name() == "r2":
        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            _s3().upload_file(str(source_path), _bucket(), key, ExtraArgs=extra_args)
        else:
            _s3().upload_file(str(source_path), _bucket(), key)
        return

    path = _local_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source_path.read_bytes())


def download_file(key: str, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if backend_name() == "r2":
        _s3().download_file(_bucket(), key, str(destination_path))
        return
    destination_path.write_bytes(_local_path(key).read_bytes())


def read_bytes(key: str) -> bytes:
    if backend_name() == "r2":
        return _s3().get_object(Bucket=_bucket(), Key=key)["Body"].read()
    return _local_path(key).read_bytes()


def write_json(key: str, value: dict[str, Any]) -> None:
    upload_bytes(
        key,
        json.dumps(value, indent=2, sort_keys=True).encode("utf-8"),
        content_type="application/json",
    )


def read_json(key: str) -> dict[str, Any]:
    return json.loads(read_bytes(key).decode("utf-8"))


def exists(key: str) -> bool:
    if backend_name() == "r2":
        try:
            _s3().head_object(Bucket=_bucket(), Key=key)
        except Exception as exc:  # noqa: BLE001 - boto3 raises generated client errors.
            if _is_not_found(exc):
                return False
            raise
        return True
    return _local_path(key).is_file()


def delete_prefix(prefix: str) -> None:
    prefix = prefix.strip("/")
    if backend_name() == "r2":
        client = _s3()
        bucket = _bucket()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                client.delete_objects(Bucket=bucket, Delete={"Objects": objects})
        return

    root = _local_path(prefix)
    if root.is_dir():
        import shutil

        shutil.rmtree(root, ignore_errors=True)


def delete_object(key: str) -> None:
    if backend_name() == "r2":
        _s3().delete_object(Bucket=_bucket(), Key=key)
        return

    path = _local_path(key)
    if path.exists():
        path.unlink()


def storage_diagnostics(run_smoke: bool = False) -> dict[str, Any]:
    backend = backend_name()
    result: dict[str, Any] = {
        "backend": backend,
        "ok": True,
        "bucket": None,
        "endpoint_host": None,
        "list_test": "not applicable",
        "head_test": "not applicable",
        "smoke_test": "not run",
        "error": None,
    }

    if backend != "r2":
        result["local_root"] = str(LOCAL_STORAGE_ROOT)
        return result

    try:
        endpoint_url = _endpoint_url()
        client = _s3()
        bucket = _bucket()
        result["bucket"] = bucket
        result["endpoint_host"] = urlparse(endpoint_url).netloc

        prefix = object_prefix()
        list_prefix = f"{prefix}/" if prefix else ""
        client.list_objects_v2(Bucket=bucket, Prefix=list_prefix, MaxKeys=1)
        result["list_test"] = "PASS"

        client.head_bucket(Bucket=bucket)
        result["head_test"] = "PASS"

        if run_smoke:
            smoke = r2_smoke_test(delete_after=True)
            result["smoke_test"] = smoke["status"]
            result["ok"] = smoke["ok"]
            result["error"] = smoke["error"]
    except Exception as exc:  # noqa: BLE001 - startup diagnostics should report all failures.
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def print_storage_doctor(run_smoke: bool = False) -> bool:
    result = storage_diagnostics(run_smoke=run_smoke)
    print("DeepSky storage doctor")
    print("======================")
    print(f"Storage backend: {result['backend']}")
    if result["backend"] == "r2":
        print(f"R2 bucket: {result.get('bucket') or '<not configured>'}")
        print(f"R2 endpoint host: {result.get('endpoint_host') or '<not configured>'}")
        print(f"R2 list test: {result['list_test']}")
        print(f"R2 head test: {result['head_test']}")
        print(f"R2 smoke test: {result['smoke_test']}")
    else:
        print(f"Local storage root: {result.get('local_root')}")

    if result["ok"]:
        print("Storage status: PASS")
    else:
        print("Storage status: FAIL")
        print(f"Storage error: {result['error']}")
    return bool(result["ok"])


def r2_smoke_test(delete_after: bool = True) -> dict[str, Any]:
    if backend_name() != "r2":
        return {
            "ok": False,
            "status": "FAIL",
            "key": None,
            "error": "R2 smoke test requires DEEPSKY_STORAGE_BACKEND=r2",
        }

    key = "/".join(segment for segment in (object_prefix(), "_healthcheck.txt") if segment)
    payload = b"deepsky-r2-healthcheck\n"
    try:
        upload_bytes(key, payload, content_type="text/plain")
        _s3().head_object(Bucket=_bucket(), Key=key)
        downloaded = read_bytes(key)
        if downloaded != payload:
            raise RuntimeError("R2 smoke test read-back content did not match")
        if delete_after:
            delete_object(key)
    except Exception as exc:  # noqa: BLE001 - smoke tests should return clear failures.
        return {
            "ok": False,
            "status": "FAIL",
            "key": key,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {"ok": True, "status": "PASS", "key": key, "error": None}


def _local_path(key: str) -> Path:
    root = LOCAL_STORAGE_ROOT.resolve()
    path = (root / key).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Refusing to access storage path outside root: {key}")
    return path


def _s3():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("R2 storage requires boto3. Add boto3 to requirements.") from exc

    return boto3.client(
        "s3",
        endpoint_url=_endpoint_url(),
        aws_access_key_id=_required_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_required_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def _bucket() -> str:
    return _required_env("R2_BUCKET")


def _endpoint_url() -> str:
    endpoint_url = _required_env("R2_ENDPOINT_URL")
    parsed = urlparse(endpoint_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(
            "R2_ENDPOINT_URL must be the Cloudflare account endpoint, "
            "for example https://<account-id>.r2.cloudflarestorage.com"
        )
    if parsed.path not in {"", "/"}:
        raise RuntimeError(
            "R2_ENDPOINT_URL must not include a bucket or path. Use "
            "https://<account-id>.r2.cloudflarestorage.com and set the "
            "bucket only in R2_BUCKET."
        )
    return endpoint_url


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    value = value.strip() if value else ""
    if not value:
        raise RuntimeError(f"{name} is required for R2 storage")
    return value


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    code = str(response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DeepSky storage diagnostics")
    parser.add_argument("--smoke-test", action="store_true", help="write/read an R2 healthcheck object")
    parser.add_argument("--keep-healthcheck", action="store_true", help="leave the healthcheck object in R2")
    args = parser.parse_args()

    if args.smoke_test:
        result = r2_smoke_test(delete_after=not args.keep_healthcheck)
        print(f"R2 smoke test: {result['status']}")
        print(f"R2 key: {result['key']}")
        if result["error"]:
            print(f"R2 error: {result['error']}")
        return 0 if result["ok"] else 1

    return 0 if print_storage_doctor(run_smoke=False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
