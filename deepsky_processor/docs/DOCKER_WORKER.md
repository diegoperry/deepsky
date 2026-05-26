# Docker Worker

The Windows host does not need Siril installed locally. DeepSky can verify
and eventually run Linux processing tools inside a worker container where Siril
is installed on the container `PATH`.

The worker verifies and runs the real local processing tools without installing
Siril on Windows.

## Why Run Siril in the Worker

Siril is a native desktop/scientific imaging tool with Linux packages and GUI
library dependencies. Running it in Docker keeps those dependencies isolated
from Windows and makes the processing environment easier to reproduce.

The verifier supports two modes:

- `local`: uses `SIRIL_CLI` from your host environment.
- `container`: ignores `SIRIL_CLI` and expects `siril-cli` on `PATH` inside the
  container.

## Build the Worker Image

From the project root:

```powershell
docker build -f Dockerfile.worker -t deepsky-worker .
```

The default worker uses `ubuntu:24.04` and installs Siril with apt:

```dockerfile
apt-get install -y siril
```

It installs CPU-only PyTorch by default so environment verification builds
faster and does not download a large CUDA wheel during the first worker build.
Python packages are installed into `/opt/deepsky-venv` instead of the system
Python environment, which avoids Ubuntu's externally-managed pip restrictions.
Use `Dockerfile.worker.cuda` later when GPU/CUDA processing is needed:

```powershell
docker build -f Dockerfile.worker.cuda -t deepsky-worker:cuda .
```

If `siril` is not available from the selected apt repositories, the Docker
build will fail at that step. In that case, use the Nix-based fallback image or
create a manual image that downloads an official portable Linux/AppImage build
and exposes a `siril-cli` executable on `PATH`.

## Nix Fallback Image

An alternate Dockerfile is provided:

```powershell
docker build -f Dockerfile.worker.nix -t deepsky-worker:nix .
```

Use this when the apt package is unavailable or you want Nix to resolve Siril
and its native dependencies.

## Run the Verifier Inside Docker

PowerShell:

```powershell
docker run --rm -it -v ${PWD}:/app deepsky-worker python3 -m deepsky_processor.pipeline.main_pipeline --doctor --mode container
```

The same image also installs `python-is-python3`, so this shorter form works:

```powershell
docker run --rm -it -v ${PWD}:/app deepsky-worker python -m deepsky_processor.pipeline.main_pipeline --doctor
```

The worker image sets:

```text
DEEPSKY_VERIFY_MODE=container
USE_XVFB=1
DEEPSKY_REQUIRE_CUDA=0
```

`USE_XVFB=1` wraps Siril commands with:

```text
xvfb-run -a siril-cli
```

Unset `USE_XVFB` if your worker image can run Siril CLI directly without a
virtual display.

CUDA is optional in the default CPU worker. To require CUDA in verification,
pass:

```powershell
-e DEEPSKY_REQUIRE_CUDA=1
```

## Mount the Project Folder

This bind mount makes your current project files available at `/app`:

```powershell
-v ${PWD}:/app
```

On Linux or macOS:

```bash
docker run --rm -it -v "$PWD":/app deepsky-worker python3 -m deepsky_processor.pipeline.main_pipeline --doctor --mode container
```

## Pass StarNet++ and SCUNet Paths

Mount folders that contain the tools/models, then pass container paths as
environment variables.

PowerShell example using the official StarNet++ Linux CLI downloaded into this
project:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -e STARNET_PATH=/app/tools/StarNet/linux/StarNetv2CLI_linux/starnet++ `
  -e SCUNET_MODEL_PATH=/app/models/SCUNet/scunet_color_real_gan.pth `
  -e SCUNET_MODEL_TYPE=official `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline --doctor --mode container
```

PowerShell example with external tool/model folders:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -v C:\AstroTools:/tools `
  -v C:\AstroModels:/models `
  -e STARNET_PATH=/tools/StarNet++/starnet++ `
  -e SCUNET_MODEL_PATH=/models/SCUNet/scunet_color_real_gan.pth `
  deepsky-worker `
  python3 -m deepsky_processor.pipeline.main_pipeline --doctor --mode container
```

Linux example:

```bash
docker run --rm -it \
  -v "$PWD":/app \
  -v "$HOME/astro-tools":/tools \
  -v "$HOME/astro-models":/models \
  -e STARNET_PATH=/tools/starnet/starnet++ \
  -e SCUNET_MODEL_PATH=/models/scunet/scunet_color_real_gan.pth \
  deepsky-worker \
  python3 -m deepsky_processor.pipeline.main_pipeline --doctor --mode container
```

## Manual Siril Test Inside Docker

```powershell
docker run --rm -it deepsky-worker xvfb-run -a siril-cli --version
```

If this fails, the container does not have a usable Siril CLI yet. Fix the
worker image before moving on to image processing.

## Manual StarNet++ Test Inside Docker

DeepSky includes a helper script for the downloaded official Linux CLI package:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  bash /app/tools/StarNet/verify_starnet_linux.sh
```

The script uses StarNet++'s bundled `rgb_test5.tif` sample and verifies that a
real `rgb_test5_s.tif` output file is created.

## First Siril Script Execution

After the verifier passes Siril, run the first real Siril execution path:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --init-job /app/deepsky_processor/jobs/test-job
```

Then execute the generated Siril script:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --siril-preprocess `
    --mode container `
    --siril-profile check `
    --workdir /app/deepsky_processor/jobs/test-job
```

The check profile deliberately starts Siril, checks script execution, and exits.
Use `lights-only` or `osc-full` when running the full pipeline with real frames.

See [NEXT_ENVIRONMENT_STEPS.md](NEXT_ENVIRONMENT_STEPS.md) for the StarNet++ and
SCUNet mount plan.
