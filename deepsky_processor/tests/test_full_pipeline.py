from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

import deepsky_processor.pipeline.full_pipeline as full_pipeline
import deepsky_processor.pipeline.single_image_pipeline as single_image_pipeline
from deepsky_processor.pipeline.config import PipelineConfig
from deepsky_processor.pipeline.full_pipeline import run_full_pipeline
from deepsky_processor.pipeline.job_layout import create_job_layout
from deepsky_processor.pipeline.target_catalog import identify_target
from deepsky_processor.pipeline.scunet_runner import run_scunet
from deepsky_processor.pipeline.single_image_pipeline import run_single_image_pipeline
from deepsky_processor.pipeline.starnet_runner import run_starnet


def _config(tmp_path: Path) -> PipelineConfig:
    starnet = tmp_path / "starnet"
    starnet.write_text("tool", encoding="utf-8")
    model = tmp_path / "model.pt"
    model.write_text("model", encoding="utf-8")
    return PipelineConfig(
        verify_mode="container",
        require_cuda=False,
        siril_cli="siril-cli",
        starnet_path=starnet,
        starnet_args=["{input}", "{output}", "256"],
        scunet_model_path=model,
        scunet_device="cpu",
        scunet_model_type="auto",
    )


def test_run_starnet_builds_real_command_with_template(
    tmp_path: Path, monkeypatch
) -> None:
    starnet = tmp_path / "starnet"
    starnet.write_text("tool", encoding="utf-8")
    input_path = tmp_path / "input.tif"
    output_path = tmp_path / "output.tif"
    input_path.write_text("image", encoding="utf-8")
    calls = {}

    def fake_run(command, check, capture_output, cwd, env, text):
        calls["command"] = command
        calls["cwd"] = cwd
        calls["env"] = env
        output_path.write_text("starless", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("deepsky_processor.pipeline.starnet_runner.subprocess.run", fake_run)

    run_starnet(
        input_path=input_path,
        output_path=output_path,
        starnet_path=starnet,
        args_template=["--input", "{input}", "--output", "{output}"],
    )

    assert calls["command"] == [
        str(starnet),
        "--input",
        str(input_path.resolve()),
        "--output",
        str(output_path.resolve()),
    ]
    assert calls["cwd"] == starnet.parent
    assert str(starnet.parent) in calls["env"]["LD_LIBRARY_PATH"]


def test_run_scunet_fails_clearly_for_invalid_model(tmp_path: Path) -> None:
    model = tmp_path / "model.pth"
    model.write_text("not torchscript", encoding="utf-8")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    input_path = tmp_path / "input.tif"
    assert cv2.imwrite(str(input_path), image)

    try:
        run_scunet(input_path, tmp_path / "out.tif", model, "cpu")
    except RuntimeError as exc:
        assert "SCUNet model could not be loaded" in str(exc)
    else:
        raise AssertionError("Expected non-TorchScript SCUNet model to fail clearly")


def test_single_image_pipeline_runs_siril_before_starnet(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.tif"
    image = np.zeros((8, 8, 3), dtype=np.uint16)
    assert cv2.imwrite(str(input_path), image)
    calls = []

    def fake_siril(script_path, working_directory, *args, **kwargs):
        calls.append("siril")
        script = script_path.read_text(encoding="utf-8")
        assert "load \"input.tif\"" in script
        assert "savetif" in script
        prepared = working_directory / "siril_prepared.tif"
        assert cv2.imwrite(str(prepared), image)
        return subprocess.CompletedProcess(["siril"], 0, "siril ok", "")

    def fake_starnet(input_path, output_path, *args, **kwargs):
        calls.append("starnet")
        assert input_path.name == "starnet_input.tif"
        assert cv2.imwrite(str(output_path), image)
        return subprocess.CompletedProcess(["starnet"], 0, "starnet ok", "")

    def fake_scunet(input_path, output_path, *args, **kwargs):
        calls.append("scunet")
        assert cv2.imwrite(str(output_path), image)
        return output_path

    def fake_compose(original_path, starless_path, denoised_path, output_path, **kwargs):
        calls.append("compose")
        assert kwargs["preserve_color_calibration"] is False
        assert kwargs["target_profile"] is None
        assert kwargs["star_source_path"].name == "siril_prepared.tif"
        assert original_path.name == "starnet_input.tif"
        assert starless_path.name == "starless.tif"
        assert denoised_path.name == "denoised.tif"
        assert cv2.imwrite(str(output_path), image)
        return output_path

    def fake_prepare(input_path, output_path):
        calls.append("prepare")
        assert input_path.name == "siril_prepared.tif"
        assert cv2.imwrite(str(output_path), image)
        return output_path

    monkeypatch.setattr(single_image_pipeline, "run_siril_script", fake_siril)
    monkeypatch.setattr(single_image_pipeline, "prepare_starnet_input_file", fake_prepare)
    monkeypatch.setattr(single_image_pipeline, "run_starnet", fake_starnet)
    monkeypatch.setattr(single_image_pipeline, "run_scunet", fake_scunet)
    monkeypatch.setattr(single_image_pipeline, "compose_deepsky_image", fake_compose)

    artifacts = run_single_image_pipeline(input_path, tmp_path / "output", _config(tmp_path), "container")

    assert calls == ["siril", "prepare", "starnet", "scunet", "compose"]
    assert artifacts.siril_prepared.exists()
    assert artifacts.starnet_input.exists()
    assert artifacts.final_image.exists()


def test_single_image_pipeline_accepts_fits_before_siril_conversion(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.fits"
    input_path.write_text("not a real fits because Siril is mocked", encoding="utf-8")
    image = np.zeros((8, 8, 3), dtype=np.uint16)

    def fake_siril(script_path, working_directory, *args, **kwargs):
        script = script_path.read_text(encoding="utf-8")
        assert "load \"input.fits\"" in script
        assert "Photometric Color Calibration skipped" in script
        prepared = working_directory / "siril_prepared.tif"
        assert cv2.imwrite(str(prepared), image)
        return subprocess.CompletedProcess(["siril"], 0, "siril ok", "")

    def fake_prepare(input_path, output_path):
        assert input_path.name == "siril_prepared.tif"
        assert cv2.imwrite(str(output_path), image)
        return output_path

    def fake_starnet(input_path, output_path, *args, **kwargs):
        assert cv2.imwrite(str(output_path), image)
        return subprocess.CompletedProcess(["starnet"], 0, "starnet ok", "")

    def fake_scunet(input_path, output_path, *args, **kwargs):
        assert cv2.imwrite(str(output_path), image)
        return output_path

    def fake_compose(original_path, starless_path, denoised_path, output_path, **kwargs):
        assert kwargs["preserve_color_calibration"] is False
        assert kwargs["target_profile"] is None
        assert kwargs["star_source_path"].name == "siril_prepared.tif"
        assert cv2.imwrite(str(output_path), image)
        return output_path

    monkeypatch.setattr(single_image_pipeline, "run_siril_script", fake_siril)
    monkeypatch.setattr(single_image_pipeline, "prepare_starnet_input_file", fake_prepare)
    monkeypatch.setattr(single_image_pipeline, "run_starnet", fake_starnet)
    monkeypatch.setattr(single_image_pipeline, "run_scunet", fake_scunet)
    monkeypatch.setattr(single_image_pipeline, "compose_deepsky_image", fake_compose)

    artifacts = run_single_image_pipeline(input_path, tmp_path / "output", _config(tmp_path), "container")

    assert artifacts.final_image.exists()


def test_single_image_pipeline_adds_siril_pcc_when_fits_has_solving_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "input.fit"
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    3",
        "CRVAL1  =        149.052582103",
        "CRVAL2  =        69.2658425006",
        "FOCALLEN=                  250",
        "XPIXSZ  =     2.90000009536743",
        "END",
    ]
    header = "".join(card.ljust(80) for card in cards).encode("ascii")
    input_path.write_bytes(header.ljust(2880, b" "))
    image = np.zeros((8, 8, 3), dtype=np.uint16)

    def fake_siril(script_path, working_directory, *args, **kwargs):
        script = script_path.read_text(encoding="utf-8")
        assert "pcc 149.052582103,69.2658425006" in script
        assert "-focal=250" in script
        assert "-pixelsize=2.90000009536743" in script
        prepared = working_directory / "siril_prepared.tif"
        assert cv2.imwrite(str(prepared), image)
        return subprocess.CompletedProcess(["siril"], 0, "siril ok", "")

    def fake_prepare(input_path, output_path):
        assert cv2.imwrite(str(output_path), image)
        return output_path

    def fake_starnet(input_path, output_path, *args, **kwargs):
        assert cv2.imwrite(str(output_path), image)
        return subprocess.CompletedProcess(["starnet"], 0, "starnet ok", "")

    def fake_scunet(input_path, output_path, *args, **kwargs):
        assert cv2.imwrite(str(output_path), image)
        return output_path

    def fake_compose(original_path, starless_path, denoised_path, output_path, **kwargs):
        assert kwargs["preserve_color_calibration"] is True
        assert kwargs["target_profile"] == "galaxy"
        assert kwargs["star_source_path"].name == "siril_prepared.tif"
        assert cv2.imwrite(str(output_path), image)
        return output_path

    monkeypatch.setattr(single_image_pipeline, "run_siril_script", fake_siril)
    monkeypatch.setattr(single_image_pipeline, "prepare_starnet_input_file", fake_prepare)
    monkeypatch.setattr(single_image_pipeline, "run_starnet", fake_starnet)
    monkeypatch.setattr(single_image_pipeline, "run_scunet", fake_scunet)
    monkeypatch.setattr(single_image_pipeline, "compose_deepsky_image", fake_compose)

    artifacts = run_single_image_pipeline(input_path, tmp_path / "output", _config(tmp_path), "container")

    assert artifacts.final_image.exists()


def test_openngc_seed_identifies_m81_from_fits_object(tmp_path: Path) -> None:
    input_path = tmp_path / "m81.fit"
    cards = [
        "SIMPLE  =                    T",
        "BITPIX  =                   16",
        "NAXIS   =                    3",
        "OBJECT  = 'M 81    '",
        "END",
    ]
    header = "".join(card.ljust(80) for card in cards).encode("ascii")
    input_path.write_bytes(header.ljust(2880, b" "))

    target = identify_target(input_path)

    assert target is not None
    assert target.name == "M 81"
    assert target.profile == "galaxy"


def test_full_pipeline_orchestrates_real_tool_stages_with_mocks(
    tmp_path: Path, monkeypatch
) -> None:
    layout = create_job_layout(tmp_path / "job")
    (layout.lights / "light_001.fit").write_text("frame", encoding="utf-8")

    def fake_siril(*args, **kwargs):
        stack = layout.output / "result.tif"
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        assert cv2.imwrite(str(stack), image)
        return subprocess.CompletedProcess(["siril"], 0, "siril ok", "")

    def fake_starnet(input_path, output_path, *args, **kwargs):
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        assert cv2.imwrite(str(output_path), image)
        return subprocess.CompletedProcess(["starnet"], 0, "starnet ok", "")

    def fake_scunet(input_path, output_path, *args, **kwargs):
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        assert cv2.imwrite(str(output_path), image)
        return output_path

    monkeypatch.setattr(full_pipeline, "run_siril_script", fake_siril)
    monkeypatch.setattr(full_pipeline, "run_starnet", fake_starnet)
    monkeypatch.setattr(full_pipeline, "run_scunet", fake_scunet)

    artifacts = run_full_pipeline(layout.root, _config(tmp_path), "container", "lights-only")

    assert artifacts.siril_stack == layout.output / "result.tif"
    assert artifacts.starless.exists()
    assert artifacts.denoised.exists()
    assert artifacts.final_image.exists()
    assert (layout.logs / "01_siril.stdout.log").read_text(encoding="utf-8") == "siril ok"
