from __future__ import annotations

import zipfile
import os

from deepsky_processor.deploy import railway_worker


def test_starnet_zip_install_copies_binary_weights_and_libraries(monkeypatch, tmp_path):
    package_dir = tmp_path / "package"
    lib_dir = package_dir / "nested" / "lib"
    lib_dir.mkdir(parents=True)
    (package_dir / "starnet2").write_bytes(b"binary")
    (package_dir / "StarNet2_weights.onnx").write_bytes(b"weights")
    (lib_dir / "libopencv_core.so.406").write_bytes(b"opencv")

    archive = tmp_path / "starnet.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in package_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(package_dir))

    target = tmp_path / "runtime" / "starnet++"
    monkeypatch.setenv("STARNET_ZIP_URL", archive.as_uri())

    railway_worker._install_starnet(target)

    assert target.read_bytes() == b"binary"
    assert (target.parent / "StarNet2_weights.onnx").read_bytes() == b"weights"
    assert (target.parent / "libopencv_core.so.406").read_bytes() == b"opencv"
    if os.name != "nt":
        assert target.stat().st_mode & 0o111


def test_missing_starnet_without_url_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("STARNET_ZIP_URL", raising=False)
    monkeypatch.delenv("STARNET_BINARY_URL", raising=False)
    monkeypatch.delenv("STARNET_WEIGHTS_URL", raising=False)

    try:
        railway_worker._install_starnet(tmp_path / "runtime" / "starnet++")
    except RuntimeError as exc:
        assert "STARNET_ZIP_URL" in str(exc)
    else:
        raise AssertionError("Expected missing StarNet source to fail")
