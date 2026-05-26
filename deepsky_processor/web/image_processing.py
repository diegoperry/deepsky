"""In-memory single-image processing for the DeepSky web MVP."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ProcessedImage:
    filename: str
    media_type: str
    content: bytes


def process_uploaded_image(content: bytes, original_filename: str) -> ProcessedImage:
    """Decode, stretch, and encode one uploaded image without persistent storage."""

    if not content:
        raise ValueError("Uploaded file is empty")

    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(
            "OpenCV could not decode this file. Try TIFF, PNG, or JPEG for this MVP."
        )

    display_image = _make_display_image(image)
    ok, output = cv2.imencode(".png", display_image)
    if not ok:
        raise RuntimeError("OpenCV failed to encode processed PNG")

    stem = original_filename.rsplit(".", 1)[0] if original_filename else "deepsky"
    return ProcessedImage(
        filename=f"{stem}_deepsky.png",
        media_type="image/png",
        content=output.tobytes(),
    )


def _make_display_image(image: np.ndarray) -> np.ndarray:
    """Create an 8-bit display image with astrophotography-friendly stretching."""

    if image.ndim == 2:
        stretched = _stretch_channel(image)
        return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        channels = cv2.split(image)
        stretched_channels = [_stretch_channel(channel) for channel in channels[:3]]
        stretched = cv2.merge(stretched_channels)
        return _enhance_color_image(stretched)

    raise ValueError(f"Unsupported image dimensions: {image.shape}")


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    channel_float = channel.astype(np.float32)
    finite = channel_float[np.isfinite(channel_float)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixel values")

    low, high = np.percentile(finite, [0.5, 99.7])
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        return np.zeros(channel.shape, dtype=np.uint8)

    normalized = np.clip((channel_float - low) / (high - low), 0.0, 1.0)
    # Gentle arcsinh-like stretch that pulls faint nebulosity forward without
    # pretending to be a calibrated astrophotography workflow.
    stretched = np.arcsinh(normalized * 8.0) / np.arcsinh(8.0)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def _enhance_color_image(image: np.ndarray) -> np.ndarray:
    denoised = cv2.bilateralFilter(image, d=5, sigmaColor=30, sigmaSpace=30)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.1)
    sharpened = cv2.addWeighted(denoised, 1.25, blurred, -0.25, 0)
    return sharpened
