from __future__ import annotations

from pathlib import Path

import pytest

import deepsky_processor.pipeline.main_pipeline as main_pipeline
import deepsky_processor.pipeline.siril_runner as siril_runner
from deepsky_processor.pipeline.job_layout import (
    create_job_layout,
    validate_job_layout,
    validate_profile_inputs,
)
from deepsky_processor.pipeline.main_pipeline import (
    CheckResult,
    main,
    print_doctor_report,
    print_verification_report,
    run_environment_verification,
)
from deepsky_processor.pipeline.opencv_runner import verify_16bit_tiff_roundtrip
from deepsky_processor.pipeline.scunet_runner import validate_scunet_model_path
from deepsky_processor.pipeline.siril_script_builder import write_preprocess_script
from deepsky_processor.pipeline.starnet_runner import validate_starnet_path


def test_verify_reports_all_required_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_check(name, check, fix):
        return CheckResult(name=name, ok=True, detail="not invoked", fix="")

    monkeypatch.setattr(main_pipeline, "_run_check", fake_run_check)

    results = run_environment_verification()
    assert [result.name for result in results] == [
        "Python version",
        "OpenCV installed",
        "PyTorch installed",
        "CUDA availability",
        "Siril CLI available",
        "StarNet++ path configured",
        "SCUNet model path configured",
        "16-bit TIFF read/write",
    ]


def test_main_returns_failure_when_a_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_pipeline,
        "run_environment_verification",
        lambda mode=None: [
            CheckResult(
                name="Siril CLI available",
                ok=False,
                detail="missing",
                fix='$env:SIRIL_CLI = "C:\\Program Files\\Siril\\bin\\siril-cli.exe"',
            )
        ],
    )

    assert main(["--verify"]) == 1


def test_main_siril_preprocess_returns_failure_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_job_layout(tmp_path)
    script = tmp_path / "script.ssf"
    script.write_text("# script\n", encoding="utf-8")

    def fail_run_siril_script(*args, **kwargs):
        raise FileNotFoundError("Siril CLI executable was not found: siril-cli")

    monkeypatch.setattr(main_pipeline, "run_siril_script", fail_run_siril_script)

    exit_code = main(
        [
            "--siril-preprocess",
            "--mode",
            "container",
            "--workdir",
            str(tmp_path),
            "--siril-script",
            str(script),
        ]
    )

    assert exit_code == 1
    assert "Siril preprocessing failed: FileNotFoundError" in capsys.readouterr().err


def test_verification_report_prints_table_and_fixes(capsys: pytest.CaptureFixture[str]) -> None:
    results = [
        CheckResult(name="Python version", ok=True, detail="3.11.3"),
        CheckResult(
            name="Siril CLI available",
            ok=False,
            detail="FileNotFoundError: missing",
            fix='$env:SIRIL_CLI = "C:\\Program Files\\Siril\\bin\\siril-cli.exe"',
        ),
    ]

    print_verification_report(results)

    output = capsys.readouterr().out
    assert "| PASS" in output
    assert "| FAIL" in output
    assert "Siril CLI available" in output
    assert "$env:SIRIL_CLI" in output


def test_doctor_report_prints_environment_values(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIRIL_CLI", "C:\\Tools\\Siril\\siril-cli.exe")
    monkeypatch.setenv("STARNET_PATH", "C:\\Tools\\StarNet++\\starnet++.exe")
    monkeypatch.setenv("SCUNET_MODEL_PATH", "C:\\Models\\SCUNet\\model.pth")

    print_doctor_report(
        [
            CheckResult(
                name="PyTorch installed",
                ok=False,
                detail="OSError: DLL initialization failed",
                fix="python -m pip uninstall -y torch torchvision torchaudio",
            )
        ]
    )

    output = capsys.readouterr().out
    assert "Detected Environment" in output
    assert "Python executable:" in output
    assert "pip executable:" in output
    assert "Platform:" in output
    assert "Current shell:" in output
    assert "C:\\Tools\\Siril\\siril-cli.exe" in output
    assert "C:\\Tools\\StarNet++\\starnet++.exe" in output
    assert "C:\\Models\\SCUNet\\model.pth" in output
    assert "STARNET_PATH status:" in output
    assert "SCUNET_MODEL_PATH status:" in output
    assert "DLL initialization failed" in output


def test_torch_import_failure_becomes_failed_check(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise OSError("DLL initialization failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    result = main_pipeline._run_check(
        "PyTorch installed",
        main_pipeline._check_torch_installed,
        "reinstall torch",
    )

    assert not result.ok
    assert "OSError: DLL initialization failed" in result.detail
    assert result.fix == "reinstall torch"


def test_cuda_can_be_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return Torch()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert main_pipeline._check_cuda_available(require_cuda=False) == (
        "CUDA is not available to PyTorch (optional for CPU worker)"
    )


def test_cuda_required_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return Torch()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        main_pipeline._check_cuda_available(require_cuda=True)


def test_siril_version_uses_real_subprocess_call_with_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    class Completed:
        stdout = "Siril 1.2.0"
        stderr = ""

    def fake_run(command, check, capture_output, text):
        calls["command"] = command
        calls["check"] = check
        calls["capture_output"] = capture_output
        calls["text"] = text
        return Completed()

    monkeypatch.setattr(siril_runner.shutil, "which", lambda command: "C:\\Siril\\siril-cli.exe")
    monkeypatch.setattr(siril_runner.subprocess, "run", fake_run)

    assert siril_runner.get_siril_version("siril-cli") == "Siril 1.2.0"
    assert calls == {
        "command": ["C:\\Siril\\siril-cli.exe", "--version"],
        "check": True,
        "capture_output": True,
        "text": True,
    }


def test_run_siril_script_uses_real_subprocess_call_with_mocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "script.ssf"
    workdir = tmp_path / "work"
    script.write_text("# script\n", encoding="utf-8")
    workdir.mkdir()
    calls = {}

    class Completed:
        stdout = "done"
        stderr = ""

    def fake_run(command, check, capture_output, cwd, text):
        calls["command"] = command
        calls["check"] = check
        calls["capture_output"] = capture_output
        calls["cwd"] = cwd
        calls["text"] = text
        return Completed()

    monkeypatch.setattr(siril_runner.shutil, "which", lambda command: "/usr/bin/siril-cli")
    monkeypatch.setattr(siril_runner.subprocess, "run", fake_run)

    result = siril_runner.run_siril_script(script, workdir, "siril-cli", {})

    assert result.stdout == "done"
    assert calls == {
        "command": ["/usr/bin/siril-cli", "-s", str(script.resolve())],
        "check": True,
        "capture_output": True,
        "cwd": workdir.resolve(),
        "text": True,
    }


def test_run_siril_script_requires_existing_paths(tmp_path: Path) -> None:
    script = tmp_path / "missing.ssf"
    workdir = tmp_path / "work"
    workdir.mkdir()

    with pytest.raises(FileNotFoundError, match="Siril script does not exist"):
        siril_runner.run_siril_script(script, workdir, "siril-cli", {})


def test_build_siril_command_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(siril_runner.shutil, "which", lambda command: f"/usr/bin/{command}")

    assert siril_runner.build_siril_command("siril-cli", ["--version"], {}) == [
        "/usr/bin/siril-cli",
        "--version",
    ]


def test_build_siril_command_with_xvfb(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {
        "siril-cli": "/usr/bin/siril-cli",
        "xvfb-run": "/usr/bin/xvfb-run",
    }
    monkeypatch.setattr(siril_runner.shutil, "which", lambda command: paths.get(command))

    assert siril_runner.build_siril_command(
        "siril-cli", ["--version"], {"USE_XVFB": "1"}
    ) == [
        "/usr/bin/xvfb-run",
        "-a",
        "/usr/bin/siril-cli",
        "--version",
    ]


def test_build_siril_command_missing_siril_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(siril_runner.shutil, "which", lambda command: None)

    with pytest.raises(FileNotFoundError, match="Siril CLI executable was not found"):
        siril_runner.build_siril_command("siril-cli", ["--version"], {})


def test_container_mode_uses_siril_cli_from_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_get_siril_version(command, env):
        seen["command"] = command
        return "Siril 1.2.0"

    def fake_run_check(name, check, fix):
        detail = check() if name == "Siril CLI available" else "not invoked"
        return CheckResult(name=name, ok=True, detail=detail)

    monkeypatch.setenv("SIRIL_CLI", "C:\\Windows\\host-siril-cli.exe")
    monkeypatch.setattr(main_pipeline, "get_siril_version", fake_get_siril_version)
    monkeypatch.setattr(main_pipeline, "_run_check", fake_run_check)

    main_pipeline.run_environment_verification("container")

    assert seen["command"] == "siril-cli"


def test_path_validators_require_configuration() -> None:
    with pytest.raises(FileNotFoundError, match="STARNET_PATH"):
        validate_starnet_path(None)
    with pytest.raises(FileNotFoundError, match="SCUNET_MODEL_PATH"):
        validate_scunet_model_path(None)


def test_path_validators_accept_existing_files(tmp_path: Path) -> None:
    starnet = tmp_path / "starnet"
    scunet = tmp_path / "scunet.pth"
    starnet.write_text("placeholder", encoding="utf-8")
    scunet.write_text("placeholder", encoding="utf-8")

    assert validate_starnet_path(starnet) == starnet
    assert validate_scunet_model_path(scunet) == scunet


def test_job_layout_creation_and_script_generation(tmp_path: Path) -> None:
    layout = create_job_layout(tmp_path / "job-001")
    script = write_preprocess_script(layout)

    assert layout.lights.is_dir()
    assert layout.darks.is_dir()
    assert layout.flats.is_dir()
    assert layout.biases.is_dir()
    assert layout.process.is_dir()
    assert layout.masters.is_dir()
    assert layout.output.is_dir()
    assert script.is_file()
    assert 'cd "' in script.read_text(encoding="utf-8")
    assert validate_job_layout(layout.root) == layout


def test_main_init_job_creates_layout(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_dir = tmp_path / "job-002"

    assert main(["--init-job", str(job_dir)]) == 0

    assert (job_dir / "lights").is_dir()
    assert (job_dir / "scripts" / "siril_preprocess.ssf").is_file()
    assert "Created DeepSky job layout" in capsys.readouterr().out


def test_main_init_job_can_generate_osc_full_script(tmp_path: Path) -> None:
    job_dir = tmp_path / "job-osc"

    assert main(["--init-job", str(job_dir), "--siril-profile", "osc-full"]) == 0

    script = (job_dir / "scripts" / "siril_preprocess.ssf").read_text(
        encoding="utf-8"
    )
    assert "Profile: osc-full" in script
    assert "calibrate light" in script
    assert "register pp_light" in script
    assert "stack r_pp_light" in script


def test_profile_input_validation_requires_lights(tmp_path: Path) -> None:
    layout = create_job_layout(tmp_path / "job-empty")

    with pytest.raises(FileNotFoundError, match="No supported frame files"):
        validate_profile_inputs(layout, "lights-only")

    (layout.lights / "light_001.fit").write_text("placeholder", encoding="utf-8")
    validate_profile_inputs(layout, "lights-only")


def test_profile_input_validation_requires_all_osc_full_folders(tmp_path: Path) -> None:
    layout = create_job_layout(tmp_path / "job-full")
    for name in ("lights", "darks", "flats", "biases"):
        directory = getattr(layout, name)
        (directory / f"{name}_001.fit").write_text("placeholder", encoding="utf-8")

    validate_profile_inputs(layout, "osc-full")


def test_16bit_tiff_roundtrip_uses_real_opencv() -> None:
    verify_16bit_tiff_roundtrip()
