# DeepSky Web Deployment

DeepSky should run as three services:

- `web`: FastAPI upload UI and temporary result server.
- `redis`: short-lived job queue.
- `worker`: Linux processing worker with Siril, StarNet++, SCUNet, and OpenCV.

The web service does not process images directly in production. It writes the upload to a temporary job folder, enqueues the job, polls status, and serves the final PNG only while the job folder exists.

## Build And Run

From the project root:

```powershell
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

## Required Local Files

The worker expects these files inside the mounted project folder:

```text
tools/StarNet/linux/StarNetv2CLI_linux/starnet++
models/SCUNet/scunet_color_real_gan.pth
```

If your paths differ, edit `docker-compose.yml`:

```yaml
STARNET_PATH: /app/tools/StarNet/linux/StarNetv2CLI_linux/starnet++
SCUNET_MODEL_PATH: /app/models/SCUNet/scunet_color_real_gan.pth
```

## Temporary Storage

Uploads and results live under:

```text
deepsky_processor/jobs/web-requests
```

The default job TTL is one hour:

```yaml
DEEPSKY_WEB_JOB_MAX_AGE_SECONDS: 3600
```

Refreshing the browser removes the in-browser preview URL. Server-side temporary job folders are cleaned when the API sees expired jobs.

## Production Notes

For a public deployment, put a reverse proxy such as Caddy, Nginx, or a managed load balancer in front of the `web` service for HTTPS.

Keep upload limits explicit at the proxy and app level. DeepSky currently targets single FITS/TIFF uploads, not multi-frame calibration sets in the web UI.

For GPU workers, build from `Dockerfile.worker.cuda`, run on a host with NVIDIA Container Toolkit, and set:

```yaml
SCUNET_DEVICE: cuda
```

The processing worker must have the real StarNet++ binary and SCUNet weights mounted or copied into the image. DeepSky does not fake those tools.
