# Web App

DeepSky includes a local web app for request-scoped single-image processing.
The default web route launches the Docker worker and runs:

```text
Siril -> StarNet++ -> SCUNet -> OpenCV
```

The multi-frame Siril stacking pipeline remains available from the CLI job
workflow. The web upload flow currently accepts one already-prepared 16-bit TIFF.

## Run

```powershell
python -m uvicorn deepsky_processor.web.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Temporary Memory Contract

The upload result is request-scoped:

- The server writes the upload to a short-lived job folder because the real
  worker tools require file paths.
- The server launches the Docker worker.
- The worker runs real Siril, StarNet++, SCUNet, and OpenCV.
- The server returns PNG bytes directly in the HTTP response.
- The server deletes the temporary job folder before the request finishes.
- The server does not create a persistent result record or saved output file.
- The browser stores the processed result as a temporary object URL.
- Refreshing the page clears the preview and download link.

Users must download the result if they want to keep it.

## Current Format Support

The worker-backed web route currently supports:

- single FITS files
- 16-bit TIFF

Siril is used as the real FITS reader/converter. The web pipeline writes a
Siril-prepared 16-bit TIFF, stretches that TIFF for StarNet++ v2, then continues
with star separation and finishing. Raw/multi-frame uploads should use the CLI
job pipeline until the web app has a multi-file job uploader.

For FITS files with RA/DEC, focal length, and pixel-size metadata, DeepSky asks
Siril to run Photometric Color Calibration before exporting the TIFF. The
current Ubuntu apt Siril package in `Dockerfile.worker` may report that it was
compiled without networking support, which prevents it from downloading the
online star catalogue needed for PCC. In that case the worker writes
`siril_color_calibration.log`, reruns Siril preparation without PCC, and does
not pretend color calibration succeeded. Use a network-enabled Siril build in
the worker image when true Siril PCC is required.

DeepSky also performs offline target identification before processing. It loads
a local OpenNGC-style CSV from `DEEPSKY_OPENNGC_CATALOG` when set, otherwise it
uses the bundled seed catalog in `deepsky_processor/data/openngc_seed.csv`.
The target is matched from FITS `OBJECT`, filename, or FITS RA/DEC and written
to `target_identification.log`. The identified profile is used to choose
galaxy/nebula behavior. When Siril PCC is unavailable, DeepSky additionally
uses unsaturated stars in the image to estimate a local white-balance correction.

## Current Processing

The default web processor performs:

- Siril single-image TIFF preparation
- Siril Photometric Color Calibration when the worker Siril build can access
  the required catalogue
- offline OpenNGC-style target identification
- local unsaturated-star white balance when photometric calibration is not
  available
- FITS-to-TIFF conversion through Siril when a FITS file is uploaded
- OpenCV working stretch to create the 16-bit TIFF StarNet++ expects
- StarNet++ star removal
- SCUNet denoising
- OpenCV background model and gradient removal
- OpenCV background neutralization and color balancing
- OpenCV chrominance denoise and HDR stretch
- Star restoration from the original-minus-starless residual
- Final composite
- PNG output

For frontend-only development, the old in-process OpenCV preview path is still
available explicitly:

```powershell
$env:DEEPSKY_WEB_PROCESSOR = "opencv"
python -m uvicorn deepsky_processor.web.app:app --host 127.0.0.1 --port 8000
```

The normal worker path expects Docker Desktop running and the `deepsky-worker`
image built.
