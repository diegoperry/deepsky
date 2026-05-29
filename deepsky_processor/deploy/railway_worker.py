"""Railway worker bootstrap for real DeepSky tool dependencies.

Railway builds from GitHub, while StarNet++ and model weights are intentionally
not committed to the repository. This bootstrap fills a mounted runtime volume
from explicit URLs when needed, then starts the RQ worker.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_QUEUE = "deepsky"
DEFAULT_REDIS_URL = "redis://redis:6379/0"


def main() -> int:
    ensure_runtime_dependencies()
    _verify_storage()
    command = sys.argv[1:] or [
        "rq",
        "worker",
        os.environ.get("DEEPSKY_QUEUE_NAME", DEFAULT_QUEUE),
        "--url",
        os.environ.get("REDIS_URL", DEFAULT_REDIS_URL),
    ]
    os.execvp(command[0], command)
    return 0


def ensure_runtime_dependencies() -> None:
    starnet_path = _path_from_env("STARNET_PATH")
    scunet_path = _path_from_env("SCUNET_MODEL_PATH")

    if starnet_path is not None and not starnet_path.exists():
        _install_starnet(starnet_path)
    if scunet_path is not None and not scunet_path.exists():
        _install_scunet(scunet_path)


def _verify_storage() -> None:
    from deepsky_processor.web import storage

    run_smoke = os.environ.get("DEEPSKY_STORAGE_SMOKE_ON_STARTUP", "0") == "1"
    ok = storage.print_storage_doctor(run_smoke=run_smoke)
    if storage.backend_name() != "local" and not ok:
        raise RuntimeError("Remote storage doctor failed; refusing to start worker.")


def _install_starnet(starnet_path: Path) -> None:
    zip_url = os.environ.get("STARNET_ZIP_URL")
    binary_url = os.environ.get("STARNET_BINARY_URL")
    weights_url = os.environ.get("STARNET_WEIGHTS_URL")

    if zip_url:
        _install_starnet_zip(zip_url, starnet_path)
    elif binary_url and weights_url:
        starnet_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(binary_url, starnet_path)
        _download_file(weights_url, starnet_path.parent / "StarNet2_weights.onnx")
    else:
        raise RuntimeError(
            "StarNet++ is missing and no download source is configured. "
            "Attach a Railway volume containing STARNET_PATH, or set "
            "STARNET_ZIP_URL to a direct ZIP URL for the official Linux CLI."
        )

    _make_executable(starnet_path)


def _install_starnet_zip(zip_url: str, starnet_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="deepsky-starnet-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "starnet.zip"
        _download_file(zip_url, archive)
        if not zipfile.is_zipfile(archive):
            raise RuntimeError(
                "STARNET_ZIP_URL did not return a ZIP file. Use a direct download "
                "URL, not an HTML redirect page."
            )
        extract_dir = tmp_path / "extract"
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)

        binary = _find_first(extract_dir, ("starnet++", "starnet2"))
        weights = _find_first(extract_dir, ("StarNet2_weights.onnx",))
        if binary is None or weights is None:
            raise RuntimeError(
                "StarNet++ ZIP is missing starnet2/starnet++ or "
                "StarNet2_weights.onnx."
            )

        starnet_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, starnet_path)
        shutil.copy2(weights, starnet_path.parent / "StarNet2_weights.onnx")

        for library in extract_dir.rglob("*.so*"):
            if library.is_file():
                shutil.copy2(library, starnet_path.parent / library.name)


def _install_scunet(scunet_path: Path) -> None:
    url = os.environ.get("SCUNET_MODEL_URL")
    if url:
        scunet_path.parent.mkdir(parents=True, exist_ok=True)
        _download_file(url, scunet_path)
        return

    if os.environ.get("DEEPSKY_AUTO_DOWNLOAD_SCUNET", "1") != "1":
        raise RuntimeError(
            "SCUNet model is missing and SCUNET_MODEL_URL is not configured."
        )

    from deepsky_processor.tools.download_scunet_model import download_model

    downloaded = download_model("real-gan", scunet_path.parent)
    if downloaded != scunet_path:
        scunet_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), scunet_path)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        with tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    tmp_path.replace(destination)


def _find_first(root: Path, names: tuple[str, ...]) -> Path | None:
    wanted = set(names)
    for path in root.rglob("*"):
        if path.is_file() and path.name in wanted:
            return path
    return None


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


if __name__ == "__main__":
    raise SystemExit(main())
