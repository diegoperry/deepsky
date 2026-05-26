"""Command-line entrypoint for the DeepSky local pipeline."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from deepsky_processor.pipeline.config import load_config
from deepsky_processor.pipeline.full_pipeline import run_full_pipeline
from deepsky_processor.pipeline.job_layout import (
    create_job_layout,
    validate_job_layout,
    validate_profile_inputs,
)
from deepsky_processor.pipeline.opencv_runner import (
    get_opencv_version,
    verify_16bit_tiff_roundtrip,
)
from deepsky_processor.pipeline.scunet_runner import validate_scunet_model_path
from deepsky_processor.pipeline.single_image_pipeline import run_single_image_pipeline
from deepsky_processor.pipeline.siril_script_builder import write_preprocess_script
from deepsky_processor.pipeline.siril_runner import get_siril_version, run_siril_script
from deepsky_processor.pipeline.starnet_runner import get_starnet_help, validate_starnet_path


DEFAULT_SIRIL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "siril_preprocess.ssf"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _run_check(name: str, check: Callable[[], str], fix: str = "") -> CheckResult:
    try:
        return CheckResult(name=name, ok=True, detail=check(), fix="")
    except Exception as exc:  # noqa: BLE001 - verification should report every failure clearly.
        return CheckResult(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}", fix=fix)


def run_environment_verification(mode: str | None = None) -> list[CheckResult]:
    """Run all local environment checks without simulating external tools."""

    config = load_config()
    verify_mode = mode or config.verify_mode
    if verify_mode not in {"local", "container"}:
        raise ValueError("Verification mode must be 'local' or 'container'")
    siril_command = "siril-cli" if verify_mode == "container" else config.siril_cli
    siril_fix = (
        "Build and run the worker image: docker build -f Dockerfile.worker -t deepsky-worker ."
        if verify_mode == "container"
        else '$env:SIRIL_CLI = "C:\\Program Files\\Siril\\bin\\siril-cli.exe"'
    )
    starnet_fix = (
        "Mount the Linux StarNet++ binary and pass -e STARNET_PATH=/tools/StarNet++/starnet++"
        if verify_mode == "container"
        else '$env:STARNET_PATH = "C:\\Tools\\StarNet++\\starnet++.exe"'
    )
    scunet_fix = (
        "Mount the SCUNet weights and pass -e SCUNET_MODEL_PATH=/models/SCUNet/scunet_color_real_gan.pth"
        if verify_mode == "container"
        else '$env:SCUNET_MODEL_PATH = "C:\\Models\\SCUNet\\scunet_color_real_gan.pth"'
    )

    checks: list[tuple[str, Callable[[], str], str]] = [
        (
            "Python version",
            lambda: f"{platform.python_version()} ({sys.executable})",
            "Use Python 3.11+ in a clean virtual environment: python -m venv .venv",
        ),
        (
            "OpenCV installed",
            lambda: f"OpenCV {get_opencv_version()}",
            "Install requirements: python -m pip install -r .\\deepsky_processor\\requirements.txt",
        ),
        (
            "PyTorch installed",
            _check_torch_installed,
            "Reinstall PyTorch in this environment: python -m pip uninstall -y torch torchvision torchaudio; python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        ),
        (
            "CUDA availability",
            lambda: _check_cuda_available(config.require_cuda),
            "Install a CUDA-capable PyTorch build and NVIDIA driver, or use CPU-only processing later if CUDA is not required.",
        ),
        (
            "Siril CLI available",
            lambda: get_siril_version(siril_command, os.environ) or "Siril CLI responded",
            siril_fix,
        ),
        (
            "StarNet++ path configured",
            lambda: str(
                validate_starnet_path(
                    config.starnet_path,
                    require_executable=verify_mode == "container",
                )
            ),
            starnet_fix,
        ),
        (
            "SCUNet model path configured",
            lambda: str(validate_scunet_model_path(config.scunet_model_path)),
            scunet_fix,
        ),
        (
            "16-bit TIFF read/write",
            lambda: f"OpenCV TIFF uint16 roundtrip succeeded ({verify_16bit_tiff_roundtrip().name})",
            "Reinstall OpenCV: python -m pip install --force-reinstall opencv-python numpy",
        ),
    ]

    return [_run_check(name, check, fix) for name, check, fix in checks]


def _check_torch_installed() -> str:
    import torch

    return f"PyTorch {torch.__version__}"


def _check_cuda_available(require_cuda: bool = True) -> str:
    import torch

    if not torch.cuda.is_available():
        if not require_cuda:
            return "CUDA is not available to PyTorch (optional for CPU worker)"
        raise RuntimeError("CUDA is not available to PyTorch")
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    return f"CUDA available: {device_count} device(s), current device: {device_name}"


def print_verification_report(results: list[CheckResult]) -> None:
    """Print a human-readable verification report."""

    print("DeepSky environment verification")
    print("=" * 35)
    print(_format_results_table(results))
    failed = [result for result in results if not result.ok]
    if failed:
        print()
        print("Fixes")
        print("-----")
        for result in failed:
            print(f"- {result.name}: {result.fix}")


def _format_results_table(results: list[CheckResult]) -> str:
    status_width = 6
    check_width = max([len("Check"), *(len(result.name) for result in results)])
    detail_width = max([len("Detail"), *(len(result.detail) for result in results)])
    border = (
        f"+-{'-' * status_width}-+-{'-' * check_width}-+-{'-' * detail_width}-+"
    )
    lines = [
        border,
        f"| {'Status'.ljust(status_width)} | {'Check'.ljust(check_width)} | {'Detail'.ljust(detail_width)} |",
        border,
    ]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(
            f"| {status.ljust(status_width)} | {result.name.ljust(check_width)} | {result.detail.ljust(detail_width)} |"
        )
    lines.append(border)
    return "\n".join(lines)


def print_doctor_report(results: list[CheckResult]) -> None:
    """Print detailed troubleshooting information for the local environment."""

    config = load_config()
    print_verification_report(results)
    print()
    print("Detected Environment")
    print("--------------------")
    print(f"Python executable: {sys.executable}")
    print(f"pip executable: {shutil.which('pip') or 'not found on PATH'}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Current shell: {_detect_shell()}")
    print(f"Verification mode: {config.verify_mode}")
    print(f"Require CUDA: {config.require_cuda}")
    print(f"USE_XVFB: {os.environ.get('USE_XVFB', '<not set>')}")
    print(f"SIRIL_CLI: {os.environ.get('SIRIL_CLI', '<not set; defaults to siril-cli>')}")
    siril_command = "siril-cli" if config.verify_mode == "container" else config.siril_cli
    print(f"Siril command for this mode: {siril_command}")
    print(f"Siril resolved: {shutil.which(siril_command) or '<not found>'}")
    print(f"STARNET_PATH: {os.environ.get('STARNET_PATH', '<not set>')}")
    print(f"STARNET_PATH status: {_path_status(os.environ.get('STARNET_PATH'))}")
    print(f"STARNET_ARGS: {os.environ.get('STARNET_ARGS', '{input} {output} 256')}")
    print(f"SCUNET_MODEL_PATH: {os.environ.get('SCUNET_MODEL_PATH', '<not set>')}")
    print(f"SCUNET_MODEL_PATH status: {_path_status(os.environ.get('SCUNET_MODEL_PATH'))}")
    print(f"SCUNET_DEVICE: {config.scunet_device}")
    print(f"SCUNET_MODEL_TYPE: {config.scunet_model_type}")

    print()
    print("Troubleshooting")
    print("---------------")
    for result in results:
        if result.ok:
            continue
        print(f"{result.name}")
        print(f"  Problem: {result.detail}")
        print(f"  Next action: {result.fix}")
    print()
    print("See also:")
    print("- deepsky_processor\\docs\\WINDOWS_SETUP.md")
    print("- deepsky_processor\\docs\\TOOL_PATHS.md")
    print("- deepsky_processor\\docs\\DOCKER_WORKER.md")
    print("- deepsky_processor\\docs\\NEXT_ENVIRONMENT_STEPS.md")


def _detect_shell() -> str:
    """Best-effort shell detection from inherited environment variables."""

    if os.environ.get("PSModulePath"):
        return "PowerShell indicators detected (PSModulePath is set)"
    comspec = os.environ.get("ComSpec")
    if comspec:
        return f"Unknown Windows shell; ComSpec={comspec}"
    shell = os.environ.get("SHELL")
    if shell:
        return shell
    return "unknown"


def _path_status(value: str | None) -> str:
    if not value:
        return "not configured"
    return "exists" if os.path.isfile(os.path.expanduser(value)) else "missing or not a file"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepsky_processor.pipeline.main_pipeline",
        description="DeepSky local astrophotography processing pipeline.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify local dependencies and configured external tools.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Verify the environment and print detailed troubleshooting information.",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "container"),
        default=None,
        help="Verification mode. Local uses SIRIL_CLI; container expects siril-cli on PATH.",
    )
    parser.add_argument(
        "--siril-preprocess",
        action="store_true",
        help="Run the configured Siril preprocessing script with the real Siril CLI.",
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the full Siril, StarNet++, SCUNet, and OpenCV pipeline.",
    )
    parser.add_argument(
        "--run-single-image",
        action="store_true",
        help="Run StarNet++, SCUNet, and OpenCV on one 16-bit TIFF.",
    )
    parser.add_argument(
        "--starnet-help",
        action="store_true",
        help="Call the configured real StarNet++ executable and print its help output.",
    )
    parser.add_argument(
        "--init-job",
        type=Path,
        default=None,
        help="Create a DeepSky job folder layout and generated Siril script.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for Siril script execution.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input image for --run-single-image.",
    )
    parser.add_argument(
        "--siril-script",
        type=Path,
        default=DEFAULT_SIRIL_SCRIPT,
        help="Path to a Siril .ssf script.",
    )
    parser.add_argument(
        "--siril-profile",
        choices=("check", "osc-full", "lights-only"),
        default="check",
        help="Generated Siril preprocessing script profile.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verify:
        results = run_environment_verification(args.mode)
        print_verification_report(results)
        return 0 if all(result.ok for result in results) else 1

    if args.doctor:
        if args.mode:
            os.environ["DEEPSKY_VERIFY_MODE"] = args.mode
        results = run_environment_verification(args.mode)
        print_doctor_report(results)
        return 0 if all(result.ok for result in results) else 1

    if args.init_job is not None:
        layout = create_job_layout(args.init_job)
        script_path = write_preprocess_script(layout, args.siril_profile)
        print(f"Created DeepSky job layout: {layout.root}")
        print(f"Generated Siril script: {script_path}")
        print(f"Siril profile: {args.siril_profile}")
        return 0

    if args.starnet_help:
        config = load_config()
        mode = args.mode or config.verify_mode
        try:
            print(
                get_starnet_help(
                    config.starnet_path,
                    require_executable=mode == "container",
                )
            )
        except Exception as exc:  # noqa: BLE001 - report external tool failures clearly.
            print(f"StarNet++ help failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.siril_preprocess:
        if args.workdir is None:
            parser.error("--siril-preprocess requires --workdir")
        layout = validate_job_layout(args.workdir)
        script_path = args.siril_script
        if args.siril_script == DEFAULT_SIRIL_SCRIPT:
            script_path = write_preprocess_script(layout, args.siril_profile)
            try:
                validate_profile_inputs(layout, args.siril_profile)
            except Exception as exc:  # noqa: BLE001 - CLI should print clean validation failures.
                print(f"Siril preprocessing input validation failed: {exc}", file=sys.stderr)
                return 1
        config = load_config()
        verify_mode = args.mode or config.verify_mode
        if verify_mode not in {"local", "container"}:
            parser.error("--mode must be local or container")
        siril_command = "siril-cli" if verify_mode == "container" else config.siril_cli
        try:
            result = run_siril_script(
                script_path=script_path,
                working_directory=args.workdir,
                command=siril_command,
                env=os.environ,
            )
        except Exception as exc:  # noqa: BLE001 - CLI should print clear tool failures.
            print(f"Siril preprocessing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        return 0

    if args.run_pipeline:
        if args.workdir is None:
            parser.error("--run-pipeline requires --workdir")
        config = load_config()
        mode = args.mode or config.verify_mode
        if mode not in {"local", "container"}:
            parser.error("--mode must be local or container")
        try:
            artifacts = run_full_pipeline(
                job_dir=args.workdir,
                config=config,
                mode=mode,
                siril_profile=args.siril_profile,
            )
        except Exception as exc:  # noqa: BLE001 - CLI should report real tool failures clearly.
            print(f"DeepSky pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("DeepSky pipeline completed")
        print(f"Siril stack: {artifacts.siril_stack}")
        print(f"Starless image: {artifacts.starless}")
        print(f"Denoised image: {artifacts.denoised}")
        print(f"Final image: {artifacts.final_image}")
        return 0

    if args.run_single_image:
        if args.input is None:
            parser.error("--run-single-image requires --input")
        if args.workdir is None:
            parser.error("--run-single-image requires --workdir")
        config = load_config()
        mode = args.mode or config.verify_mode
        if mode not in {"local", "container"}:
            parser.error("--mode must be local or container")
        try:
            artifacts = run_single_image_pipeline(
                input_path=args.input,
                output_dir=args.workdir,
                config=config,
                mode=mode,
            )
        except Exception as exc:  # noqa: BLE001 - CLI should report real tool failures clearly.
            print(f"DeepSky single-image pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if isinstance(exc, subprocess.CalledProcessError):
                if exc.stdout:
                    print("Subprocess stdout:", file=sys.stderr)
                    print(exc.stdout.strip(), file=sys.stderr)
                if exc.stderr:
                    print("Subprocess stderr:", file=sys.stderr)
                    print(exc.stderr.strip(), file=sys.stderr)
            return 1
        print("DeepSky single-image pipeline completed")
        print(f"Siril prepared image: {artifacts.siril_prepared}")
        print(f"StarNet++ input image: {artifacts.starnet_input}")
        print(f"Starless image: {artifacts.starless}")
        print(f"Denoised image: {artifacts.denoised}")
        print(f"Final image: {artifacts.final_image}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
