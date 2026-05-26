# DeepSky Processor

Local-first astrophotography processing pipeline skeleton.

This first version verifies the local/container environment and can execute a
conservative Siril script inside the worker. It does not simulate Siril,
StarNet++, SCUNet, OpenCV, or CUDA.

## Setup

```powershell
cd C:\Users\diego\Desktop\DeepSky
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\deepsky_processor\requirements.txt
```

For detailed Windows setup help, see
[WINDOWS_SETUP.md](docs/WINDOWS_SETUP.md).

For containerized Linux worker setup, see
[DOCKER_WORKER.md](docs/DOCKER_WORKER.md).

For the next external-tool setup decisions, see
[NEXT_ENVIRONMENT_STEPS.md](docs/NEXT_ENVIRONMENT_STEPS.md).

For the real Siril/StarNet++/SCUNet/OpenCV pipeline command, see
[FULL_PIPELINE.md](docs/FULL_PIPELINE.md).

Install the external tools separately:

- Siril command-line executable
- StarNet++ executable
- SCUNet model weights
- A PyTorch build compatible with your CUDA installation, if CUDA processing is required

## Configuration

The verifier reads these environment variables:

```powershell
$env:SIRIL_CLI = "siril-cli"
$env:STARNET_PATH = "C:\path\to\starnet++.exe"
$env:SCUNET_MODEL_PATH = "C:\path\to\scunet_model.pth"
```

`SIRIL_CLI` defaults to `siril-cli` when not set.

For more detail about what each tool path should point to, see
[TOOL_PATHS.md](docs/TOOL_PATHS.md).

## Verify Environment

Run:

```powershell
python -m deepsky_processor.pipeline.main_pipeline --verify
```

For detailed troubleshooting:

```powershell
python -m deepsky_processor.pipeline.main_pipeline --doctor
```

The command checks:

1. Python version
2. OpenCV installed
3. PyTorch installed
4. CUDA availability
5. Siril CLI available
6. StarNet++ path configured
7. SCUNet model path configured
8. Ability to write and read a 16-bit TIFF

The command exits with status `0` only when every check passes.

## First Job Layout

Inside the worker:

```powershell
docker run --rm -it -v ${PWD}:/app deepsky-worker python -m deepsky_processor.pipeline.main_pipeline --init-job /app/deepsky_processor/jobs/test-job
docker run --rm -it -v ${PWD}:/app deepsky-worker python -m deepsky_processor.pipeline.main_pipeline --siril-preprocess --mode container --siril-profile check --workdir /app/deepsky_processor/jobs/test-job
```

## Full Pipeline

After mounting StarNet++ and a TorchScript SCUNet model:

```powershell
docker run --rm -it -v ${PWD}:/app -v C:\DeepSkyTools:/tools -v C:\DeepSkyModels:/models:ro -e STARNET_PATH=/tools/StarNet++/starnet++ -e SCUNET_MODEL_PATH=/models/SCUNet/scunet_torchscript.pt -e SCUNET_DEVICE=cpu deepsky-worker python -m deepsky_processor.pipeline.main_pipeline --run-pipeline --mode container --siril-profile lights-only --workdir /app/deepsky_processor/jobs/my-job
```

## Tests

```powershell
pytest .\deepsky_processor\tests
```

## Temporary Web App

Run:

```powershell
python -m uvicorn deepsky_processor.web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

The web app keeps processed results in request/browser memory only. The server
does not create a result record or saved output file, and refreshing the page
removes the preview/download link unless you downloaded the image.

See [WEB_APP.md](docs/WEB_APP.md) for the current limitations and temporary
memory behavior.
