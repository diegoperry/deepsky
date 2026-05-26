"""Configuration checks for the real SCUNet model file."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def validate_scunet_model_path(path: Path | None) -> Path:
    """Validate that the SCUNet model path is configured and readable."""

    if path is None:
        raise FileNotFoundError(
            "SCUNet model path is not configured. Set SCUNET_MODEL_PATH to the "
            "SCUNet model weights file."
        )
    expanded = path.expanduser()
    if not expanded.exists():
        raise FileNotFoundError(f"SCUNet model file does not exist: {expanded}")
    if not expanded.is_file():
        raise FileNotFoundError(f"SCUNet model path is not a file: {expanded}")
    return expanded


def run_scunet(
    input_path: Path,
    output_path: Path,
    model_path: Path | None,
    device: str = "cpu",
    model_type: str = "auto",
) -> Path:
    """Run a real SCUNet model on an image."""

    model_file = validate_scunet_model_path(model_path)
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"SCUNet input image does not exist: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"SCUNet input path is not a file: {input_path}")

    import torch

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read SCUNet input image: {input_path}")

    tensor, original_dtype = _image_to_tensor(image)
    model = _load_scunet_model(model_file, device, model_type)

    model.eval()
    with torch.no_grad():
        output = model(tensor.to(device))
    if isinstance(output, (tuple, list)):
        output = output[0]

    output_image = _tensor_to_image(output.detach().cpu(), original_dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output_image):
        raise RuntimeError(f"OpenCV failed to write SCUNet output: {output_path}")
    return output_path


def _load_scunet_model(model_file: Path, device: str, model_type: str):
    import torch

    if model_type not in {"auto", "torchscript", "official"}:
        raise ValueError("SCUNET_MODEL_TYPE must be one of: auto, torchscript, official")

    if model_type in {"auto", "torchscript"}:
        try:
            model = torch.jit.load(str(model_file), map_location=device)
            return model.to(device)
        except Exception as exc:
            if model_type == "torchscript":
                raise RuntimeError(
                    "SCUNet model could not be loaded as TorchScript. "
                    f"Original error: {exc}"
                ) from exc

    if model_type in {"auto", "official"}:
        try:
            return _load_official_scunet_state_dict(model_file, device)
        except Exception as exc:
            raise RuntimeError(
                "SCUNet model could not be loaded as either TorchScript or the "
                "official cszn/SCUNet state dict. Use SCUNET_MODEL_TYPE to force "
                f"a loader. Original error: {exc}"
            ) from exc

    raise RuntimeError("Unreachable SCUNet model loader state")


def _load_official_scunet_state_dict(model_file: Path, device: str):
    import torch

    from deepsky_processor.pipeline.scunet_model import SCUNet

    try:
        checkpoint = torch.load(str(model_file), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(model_file), map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "params" in checkpoint:
            checkpoint = checkpoint["params"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Official SCUNet checkpoint did not contain a state dict")

    checkpoint = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    in_nc = _infer_input_channels(checkpoint)
    config = _infer_scunet_config(checkpoint)
    model = SCUNet(in_nc=in_nc, config=config)
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device)


def _infer_input_channels(state_dict: dict) -> int:
    weight = state_dict.get("m_head.0.weight")
    if weight is None:
        return 3
    return int(weight.shape[1])


def _infer_scunet_config(state_dict: dict) -> list[int]:
    sections = ("m_down1", "m_down2", "m_down3", "m_body", "m_up3", "m_up2", "m_up1")
    config: list[int] = []
    for section in sections:
        block_indexes = {
            int(parts[1])
            for key in state_dict
            if key.startswith(f"{section}.")
            for parts in [key.split(".")]
            if len(parts) > 2 and parts[1].isdigit() and parts[2] == "trans_block"
        }
        if not block_indexes:
            raise RuntimeError(f"Could not infer SCUNet block count for {section}")
        config.append(len(block_indexes))
    return config


def _image_to_tensor(image: np.ndarray):
    original_dtype = image.dtype
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"Unsupported SCUNet image dimensions: {image.shape}")

    scale = 65535.0 if image.dtype == np.uint16 else 255.0
    image_float = image.astype(np.float32) / scale
    tensor = np.transpose(image_float, (2, 0, 1))[None, ...]

    import torch

    return torch.from_numpy(tensor), original_dtype


def _tensor_to_image(tensor, dtype: np.dtype) -> np.ndarray:
    array = tensor.numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unsupported SCUNet output tensor shape: {array.shape}")

    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array, 0.0, 1.0)
    if dtype == np.uint16:
        output = (array * 65535.0).round().astype(np.uint16)
    else:
        output = (array * 255.0).round().astype(np.uint8)
    if output.shape[2] == 1:
        return output[:, :, 0]
    return output
