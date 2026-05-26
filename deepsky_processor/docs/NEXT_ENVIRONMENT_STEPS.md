# Next Environment Steps

DeepSky now verifies Siril inside the Linux worker container. The remaining
external paths are StarNet++ and SCUNet.

## StarNet++

Use a Linux StarNet++ CLI build inside the worker, not a Windows `.exe`.

Recommended host layout:

```text
C:\DeepSkyTools\StarNet++\starnet++
```

Container path:

```text
/tools/StarNet++/starnet++
```

Run with:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -v C:\DeepSkyTools:/tools `
  -e STARNET_PATH=/tools/StarNet++/starnet++ `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline --doctor
```

## SCUNet

SCUNet is model weights loaded by Python/PyTorch. Keep the model file on the
host and mount it read-only into the worker.

Recommended host layout:

```text
C:\DeepSkyModels\SCUNet\scunet_color_real_gan.pth
```

Container path:

```text
/models/SCUNet/scunet_color_real_gan.pth
```

Run with:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -v C:\DeepSkyModels:/models:ro `
  -e SCUNET_MODEL_PATH=/models/SCUNet/scunet_color_real_gan.pth `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline --doctor
```

## Combined Worker Verification

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -v C:\DeepSkyTools:/tools `
  -v C:\DeepSkyModels:/models:ro `
  -e STARNET_PATH=/tools/StarNet++/starnet++ `
  -e SCUNET_MODEL_PATH=/models/SCUNet/scunet_color_real_gan.pth `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline --doctor
```

The default worker is CPU-only. CUDA is optional unless explicitly required:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -e DEEPSKY_REQUIRE_CUDA=1 `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline --doctor
```

For GPU work later, build the CUDA image and run with Docker GPU access:

```powershell
docker build -f Dockerfile.worker.cuda -t deepsky-worker:cuda .
docker run --rm -it --gpus all -v ${PWD}:/app deepsky-worker:cuda python -m deepsky_processor.pipeline.main_pipeline --doctor
```

## Job Layout

Create a job layout before running Siril preprocessing:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --init-job /app/deepsky_processor/jobs/test-job
```

This creates:

```text
jobs/test-job/
  lights/
  darks/
  flats/
  biases/
  process/
  masters/
  output/
  scripts/siril_preprocess.ssf
```

Run `--init-job` inside the Docker worker for container jobs. That ensures the
generated Siril script contains `/app/...` Linux paths instead of Windows host
paths.

## Siril Profiles

DeepSky can generate three Siril script profiles:

- `check`: starts Siril, changes into the job directory, and exits. Use this for
  smoke tests.
- `lights-only`: converts, registers, and stacks frames from `lights/` without
  calibration frames.
- `osc-full`: converts biases/flats/darks/lights, creates masters, calibrates
  lights, registers calibrated lights, and stacks the result.

Generate a lights-only job script:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --init-job /app/deepsky_processor/jobs/lights-only-test `
    --siril-profile lights-only
```

Generate a full OSC preprocessing script:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --init-job /app/deepsky_processor/jobs/osc-full-test `
    --siril-profile osc-full
```

`lights-only` requires supported files in `lights/`. `osc-full` requires
supported files in `lights/`, `darks/`, `flats/`, and `biases/`.

## First Siril Execution Check

This command runs the real Siril CLI inside the worker with the current
conservative script:

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

The generated script changes Siril into the job directory and records the
expected frame folders. It does not calibrate, register, stack, stretch,
denoise, or remove stars yet.

Run a real lights-only preprocessing job after adding light frames:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --siril-preprocess `
    --mode container `
    --siril-profile lights-only `
    --workdir /app/deepsky_processor/jobs/lights-only-test
```

Run a real full OSC preprocessing job after adding all calibration folders:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --siril-preprocess `
    --mode container `
    --siril-profile osc-full `
    --workdir /app/deepsky_processor/jobs/osc-full-test
```
