# Railway Deployment

DeepSky can run on Railway as three services:

- `web`: FastAPI upload UI.
- `redis`: Railway Redis service.
- `worker`: Linux processing worker with Siril, StarNet++, SCUNet, and OpenCV.

Do not deploy only the web service. The web service queues work; the worker does
the real astrophotography processing.

## Why The Worker Needs A Runtime Volume

The GitHub repo intentionally does not include large external tool/model files:

```text
tools/StarNet/linux/StarNetv2CLI_linux/starnet++
tools/StarNet/linux/StarNetv2CLI_linux/StarNet2_weights.onnx
models/SCUNet/scunet_color_real_gan.pth
```

Railway clones GitHub during deployment, so these files must come from either:

- a Railway volume mounted into the worker, already containing the files, or
- direct download URLs provided as environment variables.

DeepSky will not fake StarNet++ or SCUNet. If those files are missing and no
download URL is configured, the worker fails clearly at startup.

## Web Service

Create a Railway service from the GitHub repo and use:

```text
Dockerfile.web
```

Set environment variables:

```text
DEEPSKY_WEB_PROCESSOR=queue
REDIS_URL=<Railway Redis private URL>
DEEPSKY_QUEUE_NAME=deepsky
DEEPSKY_WEB_JOB_MAX_AGE_SECONDS=3600
```

Railway provides `PORT` automatically. `Dockerfile.web` listens on that value.

## Redis Service

Add Railway Redis to the project. Copy its private `REDIS_URL` into both the web
and worker services.

## Worker Service

Create a second service from the same GitHub repo.

Use:

```text
Dockerfile.worker
```

Set the start command:

```bash
python -m deepsky_processor.deploy.railway_worker
```

Set environment variables:

```text
REDIS_URL=<Railway Redis private URL>
DEEPSKY_QUEUE_NAME=deepsky
DEEPSKY_VERIFY_MODE=container
DEEPSKY_REQUIRE_CUDA=0
USE_XVFB=1
STARNET_PATH=/app/runtime/tools/StarNet/starnet++
STARNET_ARGS={input} {output} 256
SCUNET_MODEL_PATH=/app/runtime/models/SCUNet/scunet_color_real_gan.pth
SCUNET_MODEL_TYPE=official
SCUNET_DEVICE=cpu
```

Attach a Railway volume to the worker at:

```text
/app/runtime
```

## Option A: Provide Direct Download URLs

If the runtime volume is empty, set:

```text
STARNET_ZIP_URL=<direct ZIP URL for the official Linux StarNet2 CLI package>
```

The URL must return the ZIP bytes directly. It cannot be a JavaScript redirect
or an HTML download page.

For SCUNet, either let DeepSky download the official `real-gan` model from the
KAIR GitHub release, or set:

```text
SCUNET_MODEL_URL=<direct URL to scunet_color_real_gan.pth>
```

The default is:

```text
DEEPSKY_AUTO_DOWNLOAD_SCUNET=1
```

## Option B: Preload The Railway Volume

If you can upload files into the volume, place them here:

```text
/app/runtime/tools/StarNet/starnet++
/app/runtime/tools/StarNet/StarNet2_weights.onnx
/app/runtime/tools/StarNet/libonnxruntime_providers_shared.so
/app/runtime/tools/StarNet/libopencv_core.so.406
/app/runtime/tools/StarNet/libopencv_imgcodecs.so.406
/app/runtime/tools/StarNet/libopencv_imgproc.so.406
/app/runtime/models/SCUNet/scunet_color_real_gan.pth
```

The StarNet++ binary must be executable:

```bash
chmod +x /app/runtime/tools/StarNet/starnet++
```

## Health Check

After deployment, open:

```text
https://<railway-web-domain>/api/health
```

The response should be:

```json
{"status":"ok","storage":"temporary-worker-job"}
```

To verify the worker, run this in the worker shell if Railway gives you one:

```bash
python -m deepsky_processor.pipeline.main_pipeline --doctor --mode container
```

StarNet++ and SCUNet should pass before expecting production-quality output.
