# Full Pipeline

DeepSky now has a real-tool pipeline orchestrator:

```text
Siril -> StarNet++ -> SCUNet -> OpenCV
```

It does not simulate unavailable tools. Each stage calls the configured tool or
model and fails clearly.

## Required Inputs

Create a job and add frames:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --init-job /app/deepsky_processor/jobs/my-job `
    --siril-profile lights-only
```

For `lights-only`, add supported frames to:

```text
deepsky_processor/jobs/my-job/lights/
```

For `osc-full`, add supported frames to:

```text
deepsky_processor/jobs/my-job/lights/
deepsky_processor/jobs/my-job/darks/
deepsky_processor/jobs/my-job/flats/
deepsky_processor/jobs/my-job/biases/
```

## StarNet++

`STARNET_PATH` must point to a real Linux StarNet++ executable inside the
container.

The default command template is:

```text
{input} {output} 256
```

For the official Linux CLI downloaded into this project:

```powershell
-e STARNET_PATH=/app/tools/StarNet/linux/StarNetv2CLI_linux/starnet++
```

If your StarNet++ CLI uses different arguments, set `STARNET_ARGS`:

```powershell
-e STARNET_ARGS="--input {input} --output {output}"
```

## SCUNet

`SCUNET_MODEL_PATH` must point to a real model file inside the container.

DeepSky supports:

- TorchScript models loadable with `torch.jit.load`
- official `cszn/SCUNet` `.pth` state dict checkpoints

Set the device with:

```powershell
-e SCUNET_DEVICE=cpu
```

or, for the CUDA worker:

```powershell
-e SCUNET_DEVICE=cuda
```

Control the loader with:

```powershell
-e SCUNET_MODEL_TYPE=auto
```

Allowed values are `auto`, `torchscript`, and `official`.

Download official SCUNet weights:

```powershell
python -m deepsky_processor.tools.download_scunet_model --model real-gan --output-dir C:\DeepSkyModels\SCUNet
```

Inside Docker, mount that folder and point to the `.pth` file:

```powershell
-v C:\DeepSkyModels:/models:ro `
-e SCUNET_MODEL_PATH=/models/SCUNet/scunet_color_real_gan.pth `
-e SCUNET_MODEL_TYPE=official
```

## Run Full Pipeline

CPU worker example:

```powershell
docker run --rm -it `
  -v ${PWD}:/app `
  -e STARNET_PATH=/app/tools/StarNet/linux/StarNetv2CLI_linux/starnet++ `
  -e SCUNET_MODEL_PATH=/app/models/SCUNet/scunet_color_real_gan.pth `
  -e SCUNET_MODEL_TYPE=official `
  -e SCUNET_DEVICE=cpu `
  deepsky-worker `
  python -m deepsky_processor.pipeline.main_pipeline `
    --run-pipeline `
    --mode container `
    --siril-profile lights-only `
    --workdir /app/deepsky_processor/jobs/my-job
```

Outputs are written under:

```text
jobs/my-job/output/
  result.*
  starless.tif
  denoised.tif
  final.png
```

Logs are written under:

```text
jobs/my-job/logs/
```
