# Tool Paths

The local pipeline uses real installed tools and model files. These variables
tell the verifier where to find them.

## SIRIL_CLI

`SIRIL_CLI` should point to the Siril command-line executable. If it is not set,
the verifier tries `siril-cli` from `PATH`.

Windows examples:

```powershell
$env:SIRIL_CLI = "C:\Program Files\Siril\bin\siril-cli.exe"
$env:SIRIL_CLI = "C:\Program Files\Siril\siril-cli.exe"
```

Linux examples:

```bash
export SIRIL_CLI=/usr/bin/siril-cli
export SIRIL_CLI=/usr/local/bin/siril-cli
```

Manual test:

```powershell
& $env:SIRIL_CLI --version
```

```bash
"$SIRIL_CLI" --version
```

## STARNET_PATH

`STARNET_PATH` should point to the StarNet++ executable file, not just the
folder containing it.

Inside Docker, this must be a Linux executable, not a Windows `.exe`.

The official Linux CLI package used by this project is available from the
StarNet download page as `StarNetv2CLI_linux.zip`. After extraction in this
workspace, the container path is:

```text
/app/tools/StarNet/linux/StarNetv2CLI_linux/starnet++
```

Windows examples:

```powershell
$env:STARNET_PATH = "C:\Tools\StarNet++\starnet++.exe"
$env:STARNET_PATH = "D:\AstroTools\StarNetv2CLI\starnet++.exe"
```

Linux examples:

```bash
export STARNET_PATH=/app/tools/StarNet/linux/StarNetv2CLI_linux/starnet++
export STARNET_PATH=/opt/starnet/starnet++
export STARNET_PATH=$HOME/tools/starnet/starnet++
```

Manual test:

```powershell
Test-Path $env:STARNET_PATH
& $env:STARNET_PATH
```

```bash
test -f "$STARNET_PATH" && echo "StarNet++ path exists"
cd "$(dirname "$STARNET_PATH")"
chmod +x ./starnet++
LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ./starnet++ rgb_test5.tif rgb_test5_s.tif 256
```

The StarNet++ v2 CLI expects 16-bit TIFF input and uses the argument form
`input.tif output.tif 256` by default.

## SCUNET_MODEL_PATH

`SCUNET_MODEL_PATH` should point to the SCUNet model weights file used by the
future denoising step. It should be a model file such as `.pth` or `.pt`, not a
directory.

DeepSky supports TorchScript models and official `cszn/SCUNet` `.pth` state dict
checkpoints. Use `SCUNET_MODEL_TYPE=auto`, `torchscript`, or `official`.

Windows examples:

```powershell
$env:SCUNET_MODEL_PATH = "C:\Models\SCUNet\scunet_color_real_gan.pth"
$env:SCUNET_MODEL_PATH = "D:\AstroModels\SCUNet\scunet_gray_25.pth"
```

Linux examples:

```bash
export SCUNET_MODEL_PATH=/opt/models/scunet/scunet_color_real_gan.pth
export SCUNET_MODEL_PATH=$HOME/models/scunet/scunet_gray_25.pth
```

Manual test:

```powershell
Test-Path $env:SCUNET_MODEL_PATH
```

```bash
test -f "$SCUNET_MODEL_PATH" && echo "SCUNet model path exists"
```

## Run the Verifier

```powershell
python -m deepsky_processor.pipeline.main_pipeline --doctor
```

```bash
python -m deepsky_processor.pipeline.main_pipeline --doctor
```
