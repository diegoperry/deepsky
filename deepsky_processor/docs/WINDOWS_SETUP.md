# Windows Setup

This project expects real local tools. The verifier does not simulate Siril,
StarNet++, SCUNet, OpenCV, PyTorch, or CUDA.

## Create a Virtual Environment

From PowerShell:

```powershell
cd C:\Users\diego\Desktop\DeepSky
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation scripts, run this once for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Install Requirements

```powershell
python -m pip install -r .\deepsky_processor\requirements.txt
```

Run the verifier:

```powershell
python -m deepsky_processor.pipeline.main_pipeline --verify
```

For detailed diagnostics:

```powershell
python -m deepsky_processor.pipeline.main_pipeline --doctor
```

## Fix a PyTorch DLL Load Failure

A failure like this usually means the current Python environment has a broken
or incompatible PyTorch install:

```text
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed
```

Reinstall PyTorch inside the active virtual environment.

For NVIDIA CUDA 12.1:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only PyTorch:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio
```

Then test PyTorch directly:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA should be available but prints `False`, update your NVIDIA driver and
install a PyTorch build that matches your CUDA runtime.

## Temporary Environment Variables

Temporary variables last only for the current PowerShell session:

```powershell
$env:SIRIL_CLI = "C:\Program Files\Siril\bin\siril-cli.exe"
$env:STARNET_PATH = "C:\Tools\StarNet++\starnet++.exe"
$env:SCUNET_MODEL_PATH = "C:\Models\SCUNet\scunet_color_real_gan.pth"
```

Run:

```powershell
python -m deepsky_processor.pipeline.main_pipeline --doctor
```

## Permanent User Environment Variables

Permanent user variables apply to future terminals:

```powershell
[Environment]::SetEnvironmentVariable("SIRIL_CLI", "C:\Program Files\Siril\bin\siril-cli.exe", "User")
[Environment]::SetEnvironmentVariable("STARNET_PATH", "C:\Tools\StarNet++\starnet++.exe", "User")
[Environment]::SetEnvironmentVariable("SCUNET_MODEL_PATH", "C:\Models\SCUNet\scunet_color_real_gan.pth", "User")
```

Close and reopen PowerShell after setting permanent variables.

Confirm values:

```powershell
echo $env:SIRIL_CLI
echo $env:STARNET_PATH
echo $env:SCUNET_MODEL_PATH
```

## Manual Tool Checks

```powershell
& $env:SIRIL_CLI --version
Test-Path $env:STARNET_PATH
Test-Path $env:SCUNET_MODEL_PATH
```
