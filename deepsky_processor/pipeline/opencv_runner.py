"""OpenCV checks used by the local pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


def get_opencv_version() -> str:
    """Return the installed OpenCV version."""

    return cv2.__version__


def verify_16bit_tiff_roundtrip() -> Path:
    """Write and read a 16-bit TIFF using real OpenCV image IO."""

    image = np.array(
        [
            [0, 1024, 4096, 65535],
            [512, 2048, 8192, 32768],
            [123, 4567, 16384, 60000],
            [42, 9000, 22000, 50000],
        ],
        dtype=np.uint16,
    )

    with tempfile.TemporaryDirectory(prefix="deepsky_verify_") as temp_dir:
        output_path = Path(temp_dir) / "roundtrip_16bit.tiff"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"OpenCV failed to write TIFF: {output_path}")

        loaded = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
        if loaded is None:
            raise RuntimeError(f"OpenCV failed to read TIFF: {output_path}")
        if loaded.dtype != np.uint16:
            raise RuntimeError(f"Expected uint16 TIFF, got {loaded.dtype}")
        if loaded.shape != image.shape:
            raise RuntimeError(f"Expected TIFF shape {image.shape}, got {loaded.shape}")
        if not np.array_equal(loaded, image):
            raise RuntimeError("16-bit TIFF roundtrip changed pixel values")

    return output_path


def stretch_image_file(input_path: Path, output_path: Path) -> Path:
    """Create a display-stretched PNG/TIFF output using real OpenCV."""

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image for final stretch: {input_path}")

    if image.ndim == 2:
        stretched = _stretch_channel(image)
        output = cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        output = cv2.merge([_stretch_channel(channel) for channel in cv2.split(image)])
    else:
        raise ValueError(f"Unsupported image dimensions for final stretch: {image.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"OpenCV failed to write final stretched image: {output_path}")
    return output_path


def prepare_starnet_input_file(input_path: Path, output_path: Path) -> Path:
    """Create the stretched 16-bit TIFF that StarNet++ v2 expects."""

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image for StarNet++ prep: {input_path}")
    if image.dtype != np.uint16:
        raise ValueError(f"StarNet++ prep expected uint16 input, got {image.dtype}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image dimensions for StarNet++ prep: {image.shape}")

    working = image.astype(np.float32) / 65535.0
    working = _remove_large_scale_gradient(working)
    working = _neutralize_background(working)
    stretched = _pre_starnet_stretch(working)
    output = np.clip(stretched * 65535.0, 0, 65535).round().astype(np.uint16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"OpenCV failed to write StarNet++ input TIFF: {output_path}")
    return output_path


def compose_deepsky_image(
    original_path: Path,
    starless_path: Path,
    denoised_path: Path,
    output_path: Path,
    preserve_color_calibration: bool = False,
    target_profile: str | None = None,
    star_source_path: Path | None = None,
) -> Path:
    """Build a finished-looking star-restored composite from real pipeline stages."""

    original = _read_float_image(original_path)
    starless = _read_float_image(starless_path)
    denoised = _read_float_image(denoised_path)
    star_source = _read_float_image(star_source_path) if star_source_path is not None else original
    original, starless, denoised, star_source = _match_shapes(original, starless, denoised, star_source)
    galaxy_profile = target_profile == "galaxy" or (target_profile is None and _is_probable_galaxy(original))
    if galaxy_profile:
        stars = _extract_star_layer(original, starless)
        star_mask = _build_star_mask(stars)
        composite = _compose_galaxy_from_stretched_base(
            original=original,
            starless=starless,
            denoised=denoised,
            star_source=star_source,
            star_mask=star_mask,
            preserve_color_calibration=preserve_color_calibration,
        )
        composite = _auto_crop_dark_edges(composite)
        output = np.clip(composite * 255.0, 0, 255).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output):
            raise RuntimeError(f"OpenCV failed to write final composite: {output_path}")
        return output_path

    nebula = _blend_detail_aware_denoise(starless, denoised)
    nebula = _remove_large_scale_gradient(nebula)
    nebula = _neutralize_background(nebula)
    if not preserve_color_calibration:
        nebula = _calibrate_galaxy_color(nebula) if galaxy_profile else _calibrate_emission_color(nebula)
    nebula = _flatten_background_floor(nebula)
    nebula = _reduce_chrominance_noise(nebula)
    nebula = _equalize_background_noise(nebula)
    nebula = _gentle_nebula_stretch(nebula)
    if preserve_color_calibration:
        nebula = _boost_preserved_color(nebula)
    else:
        nebula = _boost_galaxy_color(nebula) if galaxy_profile else _boost_nebula_color_conservative(nebula)
    nebula = _set_final_black_point(nebula)
    if galaxy_profile:
        nebula = _recover_galaxy_core_and_arms(nebula)

    stars = _extract_star_layer(original, starless)
    star_mask = _build_star_mask(stars)
    if not preserve_color_calibration:
        gains = _estimate_star_white_balance(original, star_mask)
        nebula = _apply_white_balance(nebula, gains, star_mask=None)
        stars = _apply_white_balance(stars, gains, star_mask=star_mask)
    clean_nebula = _suppress_starless_halos(nebula, star_mask)
    composite = _restore_stars(clean_nebula, stars)
    if galaxy_profile:
        composite = _finish_galaxy_scene(composite, star_mask, preserve_color_calibration)
    else:
        composite = _finish_nebula_dark_sky(composite, star_mask)
    composite = _cosmetic_cleanup(composite)
    composite = _auto_crop_dark_edges(composite)

    output = np.clip(composite * 255.0, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), output):
        raise RuntimeError(f"OpenCV failed to write final composite: {output_path}")
    return output_path


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    channel_float = channel.astype(np.float32)
    low, high = np.percentile(channel_float, [0.5, 99.7])
    if high <= low:
        low = float(np.min(channel_float))
        high = float(np.max(channel_float))
    if high <= low:
        return np.zeros(channel.shape, dtype=np.uint8)
    normalized = np.clip((channel_float - low) / (high - low), 0.0, 1.0)
    stretched = np.arcsinh(normalized * 8.0) / np.arcsinh(8.0)
    return np.clip(stretched * 255.0, 0, 255).astype(np.uint8)


def _read_float_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")
    scale = 65535.0 if image.dtype == np.uint16 else 255.0
    return np.clip(image.astype(np.float32) / scale, 0.0, 1.0)


def _match_shapes(*images: np.ndarray) -> tuple[np.ndarray, ...]:
    height = min(image.shape[0] for image in images)
    width = min(image.shape[1] for image in images)
    return tuple(image[:height, :width, :3] for image in images)


def _remove_large_scale_gradient(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    small_width = max(16, width // 64)
    small_height = max(16, height // 64)
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    background = cv2.GaussianBlur(small, (0, 0), sigmaX=5.5, sigmaY=5.5)
    background = cv2.resize(background, (width, height), interpolation=cv2.INTER_CUBIC)
    target_level = np.percentile(background.reshape(-1, 3), 20, axis=0) * 0.72
    corrected = image - background * 1.08 + target_level.reshape(1, 1, 3)
    return np.clip(corrected, 0.0, 1.0)


def _neutralize_background(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    mask = (luminance < np.percentile(luminance, 45)) & (saturation < np.percentile(saturation, 70))
    if np.count_nonzero(mask) < 128:
        mask = luminance < np.percentile(luminance, 35)
    background = np.median(image[mask], axis=0)
    neutral = np.mean(background)
    gains = neutral / np.maximum(background, 1e-4)
    gains = np.clip(gains, 0.65, 1.55)
    return np.clip(image * gains.reshape(1, 1, 3), 0.0, 1.0)


def _calibrate_emission_color(image: np.ndarray) -> np.ndarray:
    calibrated = image.copy()
    emission = _emission_mask(image)
    calibrated[:, :, 2] *= 1.12
    calibrated[:, :, 1] *= 0.96
    calibrated[:, :, 0] *= 0.9
    warm = np.clip(calibrated, 0.0, 1.0)
    mix = np.clip(emission[:, :, None] * 0.65, 0.0, 0.65)
    return np.clip(image * (1.0 - mix) + warm * mix, 0.0, 1.0)


def _calibrate_galaxy_color(image: np.ndarray) -> np.ndarray:
    calibrated = image.copy()
    luminance = _luminance(image)
    galaxy_signal = np.clip((luminance - np.percentile(luminance, 70)) / 0.35, 0.0, 1.0)
    calibrated[:, :, 2] *= 1.1
    calibrated[:, :, 1] *= 1.0
    calibrated[:, :, 0] *= 0.92
    mix = cv2.GaussianBlur(galaxy_signal.astype(np.float32), (0, 0), 5.0)
    mix = np.clip(mix[:, :, None] * 0.55, 0.0, 0.55)
    return np.clip(image * (1.0 - mix) + calibrated * mix, 0.0, 1.0)


def _flatten_background_floor(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    detail = _detail_mask(image)
    emission = _emission_mask(image)
    background_mask = (luminance < np.percentile(luminance, 58)) & (detail < 0.18) & (emission < 0.28)
    if np.count_nonzero(background_mask) < 512:
        return image

    floor = np.percentile(luminance[background_mask], 48)
    dark = np.clip((luminance - floor) / max(1.0 - floor, 1e-4), 0.0, 1.0)
    protected = np.clip((luminance - np.percentile(luminance, 70)) / 0.25, 0.0, 1.0)
    emission_protect = np.clip(emission * 1.35, 0.0, 0.92)
    mix = np.clip((1.0 - protected) * (1.0 - emission_protect) * 0.72, 0.0, 0.72)
    scaled = image * dark[:, :, None]
    return np.clip(image * (1.0 - mix[:, :, None]) + scaled * mix[:, :, None], 0.0, 1.0)


def _reduce_chrominance_noise(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    a_channel = cv2.bilateralFilter(a_channel, d=7, sigmaColor=20, sigmaSpace=9)
    b_channel = cv2.bilateralFilter(b_channel, d=7, sigmaColor=20, sigmaSpace=9)
    filtered = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    detail = _detail_mask(image)
    strength = np.clip(1.0 - detail[:, :, None] * 0.85, 0.18, 0.92)
    return np.clip(image * (1.0 - strength) + filtered * strength, 0.0, 1.0)


def _equalize_background_noise(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    detail = _detail_mask(image)
    background_mask = (luminance < np.percentile(luminance, 62)) & (detail < 0.24)
    if np.count_nonzero(background_mask) < 512:
        return image

    smooth = cv2.bilateralFilter(
        np.clip(image * 255.0, 0, 255).astype(np.uint8),
        d=9,
        sigmaColor=18,
        sigmaSpace=15,
    ).astype(np.float32) / 255.0
    mask = background_mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 9.0)
    mask = np.clip(mask[:, :, None] * 0.55, 0.0, 0.55)
    return np.clip(image * (1.0 - mask) + smooth * mask, 0.0, 1.0)


def _hdr_stretch(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    black = np.percentile(luminance, 2.8)
    white = np.percentile(luminance, 99.97)
    scaled = np.clip((image - black) / max(white - black, 1e-4), 0.0, 1.0)
    stretched = np.arcsinh(scaled * 6.5) / np.arcsinh(6.5)
    gamma = np.power(stretched, 1.18)
    return np.clip(gamma, 0.0, 1.0)


def _gentle_nebula_stretch(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    emission = _emission_mask(image)
    detail = _detail_mask(image)
    broad_luminance = cv2.GaussianBlur(luminance.astype(np.float32), (0, 0), 9.0)
    sky_mask = (emission < 0.20) & (detail < 0.20) & (broad_luminance < np.percentile(broad_luminance, 78))
    if np.count_nonzero(sky_mask) < 512:
        sky_mask = luminance < np.percentile(luminance, 55)

    stretch = _estimate_nebula_stretch(luminance, emission, detail, sky_mask)
    black = stretch["black"]
    white = stretch["white"]
    scaled = np.clip((image - black) / max(white - black, 1e-4), 0.0, 1.0)
    asinh_strength = stretch["asinh_strength"]
    stretched = np.arcsinh(scaled * asinh_strength) / np.arcsinh(asinh_strength)
    stretched = np.power(stretched, stretch["gamma"])

    signal = np.maximum(
        np.clip(emission * 1.25, 0.0, 1.0),
        _smoothstep(
            np.percentile(broad_luminance, 76),
            np.percentile(broad_luminance, 99.2),
            broad_luminance,
        ),
    )
    sky_discipline = np.clip((1.0 - signal)[:, :, None] * stretch["sky_discipline"], 0.0, stretch["sky_discipline"])
    stretched = stretched * (1.0 - sky_discipline)
    return np.clip(stretched, 0.0, 1.0)


def _estimate_nebula_stretch(
    luminance: np.ndarray,
    emission: np.ndarray,
    detail: np.ndarray,
    sky_mask: np.ndarray,
) -> dict[str, float]:
    sky_values = luminance[sky_mask]
    if sky_values.size < 512:
        sky_values = luminance.reshape(-1)

    background_median = float(np.median(sky_values))
    mad = float(np.median(np.abs(sky_values - background_median)))
    noise_sigma = max(mad * 1.4826, 1e-5)
    p90 = float(np.percentile(luminance, 90.0))
    p99 = float(np.percentile(luminance, 99.0))
    p999 = float(np.percentile(luminance, 99.9))
    bright_fraction = float(np.mean(luminance > 0.92))

    signal_span = max(p99 - background_median, 1e-5)
    highlight_span = max(p999 - background_median, 1e-5)
    snr_proxy = signal_span / noise_sigma
    contrast_proxy = max(p90 - background_median, 0.0) / max(highlight_span, 1e-5)
    emission_coverage = float(np.mean((emission > 0.18) & (detail > np.percentile(detail, 42))))

    clean_signal = np.clip((snr_proxy - 5.0) / 17.0, 0.0, 1.0)
    faint_target_need = np.clip((0.24 - p90) / 0.18, 0.0, 1.0)
    noisy_floor = np.clip((10.0 - snr_proxy) / 7.0, 0.0, 1.0)
    highlight_risk = np.clip((bright_fraction - 0.0008) / 0.006, 0.0, 1.0)
    broad_signal = np.clip(emission_coverage / 0.18, 0.0, 1.0)
    stretch_need = np.clip(
        clean_signal * 0.42
        + faint_target_need * 0.34
        + broad_signal * 0.34
        - noisy_floor * 0.35
        - highlight_risk * 0.25,
        0.0,
        1.0,
    )

    black = background_median - noise_sigma * (0.65 + 0.35 * stretch_need)
    black = max(0.0, min(black, float(np.percentile(sky_values, 46))))
    white_percentile = 99.82 + stretch_need * 0.14 - highlight_risk * 0.08
    white = float(np.percentile(luminance, np.clip(white_percentile, 99.70, 99.96)))
    white = max(white, black + 1e-4)

    return {
        "black": black,
        "white": white,
        "asinh_strength": float(1.75 + stretch_need * 5.40 - noisy_floor * 0.35),
        "gamma": float(1.30 - stretch_need * 0.58 + noisy_floor * 0.16 + contrast_proxy * 0.04),
        "sky_discipline": float(0.28 - stretch_need * 0.16 + noisy_floor * 0.12),
    }


def _pre_starnet_stretch(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    black = np.percentile(luminance, 0.08)
    white = np.percentile(luminance, 99.82)
    scaled = np.clip((image - black) / max(white - black, 1e-4), 0.0, 1.0)
    stretched = np.arcsinh(scaled * 18.0) / np.arcsinh(18.0)
    return np.clip(np.power(stretched, 0.62), 0.0, 1.0)


def _boost_nebula_color(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    luminance = _luminance(image)
    emission = _emission_mask(image)
    boost = np.interp(luminance, [0.08, 0.55, 0.95], [1.08, 1.26, 0.98])
    boost += emission * 0.24
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * boost, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.84 + emission * 0.16), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def _boost_nebula_color_conservative(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    luminance = _luminance(image)
    emission = _emission_mask(image)
    signal = _smoothstep(np.percentile(luminance, 72), np.percentile(luminance, 99.2), luminance)
    saturation_boost = 1.0 + emission * 0.16 + signal * 0.06
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.78 + emission * 0.18 + signal * 0.04), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def _boost_galaxy_color(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    luminance = _luminance(image)
    galaxy_signal = np.clip((luminance - np.percentile(luminance, 65)) / 0.42, 0.0, 1.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.08 + galaxy_signal * 0.55), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.9 + galaxy_signal * 0.08), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def _boost_preserved_color(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    luminance = _luminance(image)
    signal = np.clip((luminance - np.percentile(luminance, 55)) / 0.45, 0.0, 1.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.04 + signal * 0.16), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


def _estimate_star_white_balance(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    star_pixels = star_mask > 0.35
    if np.count_nonzero(star_pixels) < 24:
        return np.ones(3, dtype=np.float32)

    luminance = _luminance(image)
    star_luminance = luminance[star_pixels]
    low = np.percentile(star_luminance, 25)
    high = np.percentile(star_luminance, 92)
    unsaturated = star_pixels & (luminance >= low) & (luminance <= high)
    pixels = image[unsaturated]
    if pixels.shape[0] < 24:
        return np.ones(3, dtype=np.float32)

    channel_max = np.max(pixels, axis=1)
    channel_min = np.min(pixels, axis=1)
    color_spread = (channel_max - channel_min) / np.maximum(channel_max, 1e-4)
    pixels = pixels[color_spread < np.percentile(color_spread, 70)]
    if pixels.shape[0] < 24:
        return np.ones(3, dtype=np.float32)

    measured = np.median(pixels, axis=0)
    neutral = float(np.mean(measured))
    gains = neutral / np.maximum(measured, 1e-4)
    gains = gains / np.mean(gains)
    return np.clip(gains, 0.78, 1.22).astype(np.float32)


def _apply_white_balance(
    image: np.ndarray,
    gains: np.ndarray,
    star_mask: np.ndarray | None,
) -> np.ndarray:
    if np.allclose(gains, 1.0):
        return image
    balanced = np.clip(image * gains.reshape(1, 1, 3), 0.0, 1.0)
    if star_mask is None:
        luminance = _luminance(image)
        signal = np.clip((luminance - np.percentile(luminance, 20)) / 0.55, 0.0, 1.0)
        mix = np.clip(signal[:, :, None] * 0.72, 0.0, 0.72)
    else:
        mix = np.clip(star_mask[:, :, None] * 0.9, 0.0, 0.9)
    return np.clip(image * (1.0 - mix) + balanced * mix, 0.0, 1.0)


def _compose_galaxy_from_stretched_base(
    original: np.ndarray,
    starless: np.ndarray,
    denoised: np.ndarray,
    star_source: np.ndarray,
    star_mask: np.ndarray,
    preserve_color_calibration: bool,
) -> np.ndarray:
    base = _build_galaxy_gray_anchor(original, starless, denoised, star_mask, preserve_color_calibration)
    finished = _finish_galaxy_gray_anchor(base, star_mask)
    finished = _clean_primary_galaxy_object(finished)
    finished = _restore_galaxy_luminance_gradient(finished, original, star_mask)
    real_stars = _make_real_residual_star_layer(_extract_star_layer(original, starless), star_source=original)
    composite = np.clip(finished + real_stars * 1.35 * (1.0 - finished * 0.08), 0.0, 1.0)
    composite = _finalize_enhanced_black_sky(composite, finished, real_stars)
    return _minimize_restored_star_halos(composite, original, starless, finished)


def _build_galaxy_gray_anchor(
    original: np.ndarray,
    starless: np.ndarray,
    denoised: np.ndarray,
    star_mask: np.ndarray,
    preserve_color_calibration: bool,
) -> np.ndarray:
    base = np.clip(starless * 0.25 + denoised * 0.75, 0.0, 1.0)
    galaxy_signal = _galaxy_signal_from_original(base, star_mask)
    protected = np.clip(np.maximum(galaxy_signal, star_mask), 0.0, 1.0)

    base = _neutralize_galaxy_background_color(base, star_mask)
    base = _reduce_galaxy_background_chroma_noise(base, protected)

    base = _denoise_galaxy_scene_without_repainting(base, protected)
    base = _enhance_galaxy_luminance_structure(base, galaxy_signal)
    base = _heal_background_star_removal_scars(base, original, starless, star_mask, galaxy_signal)

    return np.clip(base, 0.0, 1.0)


def _finish_galaxy_gray_anchor(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    smoothed_luminance = cv2.GaussianBlur(luminance, (0, 0), 10.0)
    galaxy_signal = _smoothstep(
        np.percentile(smoothed_luminance, 82),
        np.percentile(smoothed_luminance, 99.7),
        smoothed_luminance,
    )
    galaxy_signal = cv2.GaussianBlur(galaxy_signal.astype(np.float32), (0, 0), 18.0)
    core = _smoothstep(
        np.percentile(smoothed_luminance, 97.3),
        np.percentile(smoothed_luminance, 99.9),
        smoothed_luminance,
    )
    core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 7.0)

    sky_mask = (galaxy_signal < 0.05) & (star_mask < 0.05)
    if np.count_nonzero(sky_mask) < 512:
        sky_mask = (luminance < np.percentile(luminance, 62)) & (star_mask < 0.05)
    sky_median = np.median(image[sky_mask], axis=0)

    residual_gradient = cv2.GaussianBlur(image - sky_median.reshape(1, 1, 3), (0, 0), 110.0)
    image = (image - sky_median.reshape(1, 1, 3)) * 2.25 + 0.026
    image = image - residual_gradient * 0.72 * (1.0 - galaxy_signal[:, :, None] * 0.75)
    image = np.clip(image, 0.0, 1.0)

    luminance = _luminance(image)
    core_peak = _smoothstep(
        np.percentile(luminance, 98.2),
        np.percentile(luminance, 99.98),
        luminance,
    )
    inner_disk = np.clip(galaxy_signal * (1.0 - core_peak * 0.35), 0.0, 1.0)
    lifted_luminance = np.clip(
        luminance
        + inner_disk * np.power(np.clip(luminance, 0.0, 1.0), 0.72) * 0.075
        + core_peak * np.power(np.clip(luminance, 0.0, 1.0), 0.58) * 0.16,
        0.0,
        1.0,
    )
    highlight_guard = _smoothstep(0.82, 0.98, luminance)
    lifted_luminance = luminance * highlight_guard + lifted_luminance * (1.0 - highlight_guard)
    ratio = np.clip(lifted_luminance / np.maximum(luminance, 1e-5), 0.72, 2.25)
    image = np.clip(image * ratio[:, :, None], 0.0, 1.0)

    image = _smooth_galaxy_background_for_final(image, galaxy_signal, star_mask)
    image = _warm_galaxy_final_color(image, galaxy_signal, core, star_mask)
    image = _neutralize_green_star_halos(image, star_mask, galaxy_signal)
    image = _suppress_galaxy_background_smudges(image, star_mask, galaxy_signal)
    image = _polish_galaxy_background_grain(image, star_mask, galaxy_signal)
    image = _genai_like_background_cleanup(image, star_mask, galaxy_signal)
    return np.clip(image, 0.0, 1.0)


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(value, dtype=np.float32)
    normalized = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (normalized * normalized * (3.0 - 2.0 * normalized)).astype(np.float32)


def _enhance_galaxy_luminance_structure(image: np.ndarray, galaxy_signal: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    denoised_luminance = cv2.fastNlMeansDenoising(
        np.clip(luminance * 255.0, 0, 255).astype(np.uint8),
        None,
        h=6,
        templateWindowSize=7,
        searchWindowSize=21,
    ).astype(np.float32) / 255.0
    middle_scale = cv2.GaussianBlur(denoised_luminance, (0, 0), 3.0)
    large_scale = cv2.GaussianBlur(denoised_luminance, (0, 0), 18.0)
    arms = cv2.GaussianBlur(middle_scale - large_scale, (0, 0), 1.8)
    enhanced_luminance = np.clip(luminance + arms * galaxy_signal * 0.32, 0.0, 1.0)
    ratio = np.clip(enhanced_luminance / np.maximum(luminance, 1e-5), 0.55, 1.55)
    return np.clip(image * ratio[:, :, None], 0.0, 1.0)


def _rebuild_point_stars(
    original: np.ndarray,
    starless: np.ndarray,
    star_source: np.ndarray | None = None,
    render_base: np.ndarray | None = None,
) -> np.ndarray:
    star_source = original if star_source is None else star_source
    residual = np.clip(original - starless, 0.0, 1.0)
    residual_luminance = _luminance(residual)
    small_scale = cv2.GaussianBlur(residual_luminance, (0, 0), 0.55)
    broad_scale = cv2.GaussianBlur(residual_luminance, (0, 0), 3.5)
    residual_high_pass = np.clip(small_scale - broad_scale, 0.0, 1.0)
    local_maxima = small_scale >= cv2.dilate(small_scale, np.ones((3, 3), np.uint8)) - 1e-7

    protection_source = render_base if render_base is not None else starless
    protection_luminance = cv2.GaussianBlur(_luminance(protection_source).astype(np.float32), (0, 0), 10.0)
    galaxy_protect = _smoothstep(
        np.percentile(protection_luminance, 82),
        np.percentile(protection_luminance, 99.6),
        protection_luminance,
    )

    threshold = max(float(np.percentile(residual_high_pass, 99.05)), 0.0014)
    candidate_pixels = np.argwhere((residual_high_pass > threshold) & local_maxima & (galaxy_protect < 0.72))
    height, width = residual_luminance.shape
    candidates: list[tuple[float, int, int, float, float, float, float]] = []

    for y, x in candidate_pixels:
        y1, y2 = max(0, y - 5), min(height, y + 6)
        x1, x2 = max(0, x - 5), min(width, x + 6)
        patch = residual_luminance[y1:y2, x1:x2]
        patch_y, patch_x = np.mgrid[y1:y2, x1:x2]
        distance = np.sqrt((patch_x - x) ** 2 + (patch_y - y) ** 2)
        annulus = patch[(distance > 2.0) & (distance < 5.5)]
        if annulus.size < 8:
            continue

        peak = float(residual_luminance[y, x])
        local_background = float(np.median(annulus))
        contrast = peak - local_background
        if contrast < 0.0014 or peak < 0.0025:
            continue

        near_core = float(np.median(patch[distance < 1.6])) if np.any(distance < 1.6) else peak
        far_field = (
            float(np.median(patch[(distance > 2.2) & (distance < 4.5)]))
            if np.any((distance > 2.2) & (distance < 4.5))
            else local_background
        )
        if near_core < far_field * 1.25 + 0.0005:
            continue

        footprint = (patch > local_background + contrast * 0.23) & (distance < 5.5)
        footprint_area = float(np.count_nonzero(footprint))
        if footprint_area <= 0:
            continue
        weights = np.clip(patch[footprint] - local_background, 0.0, 1.0)
        measured_radius = (
            float(np.sqrt(np.average(distance[footprint] ** 2, weights=weights)))
            if float(np.sum(weights)) > 1e-7
            else float(np.sqrt(footprint_area / np.pi))
        )
        local_flux = float(np.sum(weights))
        score = contrast + peak * 0.22 + float(residual_high_pass[y, x])
        candidates.append((score, int(y), int(x), contrast, peak, measured_radius, local_flux))

    if not candidates:
        return np.zeros_like(original, dtype=np.float32)

    contrast_cutoff = np.percentile([candidate[3] for candidate in candidates], 82)
    coordinates = np.array([[candidate[2], candidate[1]] for candidate in candidates], dtype=np.float32)
    filtered_candidates: list[tuple[float, int, int, float, float, float, float]] = []
    for candidate in candidates:
        _, y, x, contrast, _, _, _ = candidate
        distances = (coordinates[:, 0] - x) ** 2 + (coordinates[:, 1] - y) ** 2
        neighbor_count = int(np.count_nonzero(distances < 18 * 18)) - 1
        if neighbor_count > 3 and contrast < contrast_cutoff:
            continue
        filtered_candidates.append(candidate)

    filtered_candidates.sort(reverse=True)
    selected: list[tuple[int, int, float, float, float, float]] = []
    occupancy = np.zeros_like(residual_luminance, dtype=np.uint8)
    for _, y, x, contrast, peak, measured_radius, local_flux in filtered_candidates:
        if occupancy[y, x]:
            continue
        selected.append((y, x, contrast, peak, measured_radius, local_flux))
        exclusion_radius = int(np.clip(round(measured_radius * 1.35 + 2.0), 2, 7))
        cv2.circle(occupancy, (x, y), exclusion_radius, 1, -1)
        if len(selected) >= 650:
            break

    yy, xx = np.mgrid[0:height, 0:width]
    stars = np.zeros_like(original, dtype=np.float32)

    contrasts = np.array([star[2] for star in selected], dtype=np.float32)
    fluxes = np.array([star[5] for star in selected], dtype=np.float32)
    contrast_low = float(np.percentile(contrasts, 18))
    contrast_high = float(np.percentile(contrasts, 98))
    flux_low = float(np.percentile(fluxes, 18))
    flux_high = float(np.percentile(fluxes, 98))

    for y, x, contrast, peak, measured_radius, local_flux in selected:
        color_radius = int(np.clip(round(measured_radius) + 2, 2, 6))
        patch = star_source[
            max(0, y - color_radius) : min(height, y + color_radius + 1),
            max(0, x - color_radius) : min(width, x + color_radius + 1),
        ]
        color = np.percentile(patch.reshape(-1, 3), 94, axis=0)
        color_mean = max(float(np.mean(color)), 1e-5)
        color = color / color_mean
        neutral_star = np.array([0.95, 0.94, 1.0], dtype=np.float32)
        color = np.clip(neutral_star * 0.70 + color.astype(np.float32) * 0.30, 0.76, 1.20)

        brightness_rank = float(_smoothstep(contrast_low, contrast_high, np.array(contrast)).item())
        flux_rank = float(_smoothstep(flux_low, flux_high, np.array(local_flux)).item())
        star_rank = np.clip(brightness_rank * 0.66 + flux_rank * 0.34, 0.0, 1.0)
        sigma = float(np.clip(measured_radius * (0.46 + star_rank * 0.22), 0.32, 2.35))
        amplitude = float(np.clip((contrast * 6.2 + peak * 0.58) * (1.12 + star_rank * 1.18), 0.045, 1.0))
        radius = int(np.clip(np.ceil(max(measured_radius * 2.25, sigma * (3.2 + star_rank * 1.9))), 2, 12))
        ys = slice(max(0, y - radius), min(height, y + radius + 1))
        xs = slice(max(0, x - radius), min(width, x + radius + 1))
        core = np.exp(-((xx[ys, xs] - x) ** 2 + (yy[ys, xs] - y) ** 2) / (2.0 * sigma * sigma)).astype(
            np.float32
        )
        halo_sigma = max(sigma * (1.30 + star_rank * 0.85), measured_radius * 0.92)
        halo = np.exp(
            -((xx[ys, xs] - x) ** 2 + (yy[ys, xs] - y) ** 2) / (2.0 * halo_sigma * halo_sigma)
        ).astype(np.float32)
        halo_strength = 0.008 + star_rank * 0.075
        stars[ys, xs] += (core + halo * halo_strength)[:, :, None] * amplitude * color.reshape(1, 1, 3)

    return np.clip(stars, 0.0, 1.0)


def _make_display_star_layer(stars: np.ndarray) -> np.ndarray:
    star_luminance = _luminance(stars)
    if float(np.max(star_luminance)) <= 0.0:
        return stars

    positive = star_luminance[star_luminance > 0.0003]
    star_rank = _smoothstep(
        max(float(np.percentile(positive, 45)), 0.001) if positive.size else 0.001,
        max(float(np.percentile(positive, 99.2)), 0.008) if positive.size else 0.008,
        star_luminance,
    )
    core = np.clip(stars * (1.12 + star_rank[:, :, None] * 0.34), 0.0, 1.0)
    small_glow = cv2.GaussianBlur(stars, (0, 0), 0.42)
    soft_glow = cv2.GaussianBlur(stars, (0, 0), 1.05)
    display = np.clip(
        core + small_glow * (0.10 + star_rank[:, :, None] * 0.22) + soft_glow * star_rank[:, :, None] * 0.10,
        0.0,
        1.0,
    )

    display_luminance = _luminance(display)
    bright_star = _smoothstep(
        max(float(np.percentile(star_luminance[star_luminance > 0.0003], 72)), 0.002)
        if np.any(star_luminance > 0.0003)
        else 0.002,
        max(float(np.percentile(star_luminance[star_luminance > 0.0003], 98)), 0.006)
        if np.any(star_luminance > 0.0003)
        else 0.006,
        star_luminance,
    )
    white_point = np.repeat(display_luminance[:, :, None], 3, axis=2)
    display = np.clip(display * (1.0 - bright_star[:, :, None] * 0.18) + white_point * bright_star[:, :, None] * 0.18, 0.0, 1.0)
    return display


def _make_real_residual_star_layer(stars: np.ndarray, star_source: np.ndarray | None = None) -> np.ndarray:
    """Restore stars from source pixels using the StarNet residual footprint."""

    star_luminance = _luminance(stars).astype(np.float32)
    if float(np.max(star_luminance)) <= 0.0:
        return stars

    active = star_luminance[star_luminance > max(float(np.percentile(star_luminance, 70)), 0.00012)]
    if active.size == 0:
        return stars

    star_seed = _smoothstep(
        max(float(np.percentile(active, 14)), 0.00015),
        max(float(np.percentile(active, 99.65)), 0.003),
        star_luminance,
    )
    star_footprint = (star_seed > 0.10).astype(np.uint8)
    star_footprint = cv2.morphologyEx(star_footprint, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    star_footprint = cv2.dilate(star_footprint, np.ones((3, 3), np.uint8), iterations=1)
    star_mask = np.maximum(np.clip(star_seed, 0.0, 1.0).astype(np.float32), star_footprint.astype(np.float32) * 0.72)
    star_mask = cv2.GaussianBlur(star_mask, (0, 0), 0.54)

    source = stars if star_source is None else star_source
    source_luminance_full = _luminance(source).astype(np.float32)
    background_pixels = source_luminance_full[star_mask < 0.03]
    black = float(np.percentile(background_pixels, 42)) if background_pixels.size else float(np.percentile(source_luminance_full, 20))
    source_signal = np.clip(source - black * 0.92, 0.0, 1.0)
    source_luminance = _luminance(source_signal).astype(np.float32)
    source_luminance = np.clip(np.maximum(source_luminance, star_luminance * 0.82), 0.0, 1.0)

    source_color = np.divide(
        source_signal,
        np.maximum(source_luminance[:, :, None], 1e-6),
        out=np.zeros_like(source_signal),
        where=source_luminance[:, :, None] > 1e-6,
    )
    source_color = np.clip(source_color, 0.78, 1.22)
    warm_neutral = np.array([1.0, 0.93, 0.84], dtype=np.float32).reshape(1, 1, 3)
    color = np.clip(source_color * 0.38 + warm_neutral * 0.62, 0.0, 1.28)

    restored_color = np.clip(np.repeat(source_luminance[:, :, None], 3, axis=2) * color, 0.0, 1.0)
    restored_color = cv2.GaussianBlur(restored_color, (0, 0), 0.46)
    restored = np.clip(restored_color * star_mask[:, :, None], 0.0, 1.0)
    return restored


def _heal_background_star_removal_scars(
    image: np.ndarray,
    original: np.ndarray,
    starless: np.ndarray,
    star_mask: np.ndarray,
    galaxy_signal: np.ndarray,
) -> np.ndarray:
    residual = np.clip(original - starless, 0.0, 1.0)
    residual_luminance = _luminance(residual)
    threshold = max(float(np.percentile(residual_luminance, 99.05)), 0.016)
    candidates = (residual_luminance > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    star_regions = np.zeros_like(candidates)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if 1 <= area <= 180:
            star_regions[labels == label] = 1
    star_regions = cv2.dilate(star_regions, np.ones((5, 5), np.uint8), iterations=1)
    object_protect = cv2.GaussianBlur(np.clip(galaxy_signal * 2.2, 0.0, 1.0).astype(np.float32), (0, 0), 6.0)
    star_regions = (star_regions * (1.0 - object_protect)).astype(np.uint8)
    point_scar_mask = cv2.GaussianBlur(star_regions.astype(np.float32), (0, 0), 2.2)
    broad_scar_mask = cv2.GaussianBlur(star_mask.astype(np.float32), (0, 0), 2.8)
    scar_mask = np.maximum(point_scar_mask * 0.55, broad_scar_mask * 0.72)
    scar_mask = np.clip(scar_mask * (1.0 - object_protect), 0.0, 0.72)

    if float(np.max(scar_mask)) <= 0.0:
        return image

    source = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    replacement = cv2.medianBlur(source, 15).astype(np.float32) / 255.0
    replacement = cv2.bilateralFilter(
        np.clip(replacement * 255.0, 0, 255).astype(np.uint8),
        d=0,
        sigmaColor=8,
        sigmaSpace=16,
    ).astype(np.float32) / 255.0
    return np.clip(image * (1.0 - scar_mask[:, :, None]) + replacement * scar_mask[:, :, None], 0.0, 1.0)


def _smooth_galaxy_background_for_final(
    image: np.ndarray,
    galaxy_signal: np.ndarray,
    star_mask: np.ndarray,
) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(
        np.clip(image * 255.0, 0, 255).astype(np.uint8),
        None,
        h=7,
        hColor=8,
        templateWindowSize=7,
        searchWindowSize=21,
    ).astype(np.float32) / 255.0
    mix = np.clip((1.0 - galaxy_signal) * 0.45 * (1.0 - star_mask), 0.0, 0.45)
    return np.clip(image * (1.0 - mix[:, :, None]) + denoised * mix[:, :, None], 0.0, 1.0)


def _warm_galaxy_final_color(
    image: np.ndarray,
    galaxy_signal: np.ndarray,
    core: np.ndarray,
    star_mask: np.ndarray,
) -> np.ndarray:
    luminance = _luminance(image)
    sky_mask = (galaxy_signal < 0.05) & (star_mask < 0.06) & (luminance < np.percentile(luminance, 82))
    if np.count_nonzero(sky_mask) > 512:
        background = np.median(image[sky_mask], axis=0)
        neutral = float(np.mean(background))
        image = np.clip(image * (neutral / np.maximum(background, 1e-4)).reshape(1, 1, 3), 0.0, 1.0)

    galaxy_mix = galaxy_signal[:, :, None]
    warmed = image.copy()
    warmed[:, :, 0] *= 1.0 - 0.36 * galaxy_mix[:, :, 0]
    warmed[:, :, 1] *= 1.0 + 0.16 * galaxy_mix[:, :, 0]
    warmed[:, :, 2] *= 1.0 + 0.08 * galaxy_mix[:, :, 0]
    image = np.clip(image * (1.0 - galaxy_mix * 0.82) + warmed * galaxy_mix * 0.82, 0.0, 1.0)

    cream = np.zeros_like(image)
    cream[:, :, 0] = 0.78
    cream[:, :, 1] = 0.86
    cream[:, :, 2] = 1.0
    image_luminance = _luminance(image)[:, :, None]
    core_mix = np.clip(core[:, :, None] * 0.28, 0.0, 0.28)
    image = np.clip(image * (1.0 - core_mix) + cream * image_luminance * core_mix, 0.0, 1.0)
    core_luminance = _luminance(image)
    white_core = _smoothstep(
        np.percentile(core_luminance, 98.25),
        np.percentile(core_luminance, 99.92),
        core_luminance,
    )
    white_core = cv2.GaussianBlur((white_core * np.clip(core, 0.0, 1.0)).astype(np.float32), (0, 0), 2.8)
    neutral_core = np.repeat(core_luminance[:, :, None], 3, axis=2)
    image = np.clip(image * (1.0 - white_core[:, :, None] * 0.72) + neutral_core * white_core[:, :, None] * 0.72, 0.0, 1.0)

    hsv = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.50 + galaxy_signal * 0.16
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    neutral = np.repeat(_luminance(image)[:, :, None], 3, axis=2)
    sky_mix = np.clip((1.0 - galaxy_signal) * (1.0 - star_mask) * 0.22, 0.0, 0.22)
    return np.clip(image * (1.0 - sky_mix[:, :, None]) + neutral * sky_mix[:, :, None], 0.0, 1.0)


def _neutralize_green_star_halos(
    image: np.ndarray,
    star_mask: np.ndarray,
    galaxy_signal: np.ndarray,
) -> np.ndarray:
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    neutral_green = (blue + red) * 0.5
    green_excess = np.clip(green - neutral_green, 0.0, 1.0)
    halo_mask = cv2.GaussianBlur(star_mask.astype(np.float32), (0, 0), 1.8)
    halo_mask = np.clip(halo_mask * (1.0 - galaxy_signal * 0.35), 0.0, 1.0)

    corrected = image.copy()
    corrected[:, :, 1] = np.clip(green - green_excess * 0.92 * halo_mask, 0.0, 1.0)
    corrected[:, :, 0] = np.clip(corrected[:, :, 0] + green_excess * 0.18 * halo_mask, 0.0, 1.0)
    corrected[:, :, 2] = np.clip(corrected[:, :, 2] + green_excess * 0.22 * halo_mask, 0.0, 1.0)

    luminance = _luminance(corrected)
    star_core = np.clip(star_mask * 1.35, 0.0, 1.0)
    white_star = np.repeat(luminance[:, :, None], 3, axis=2)
    return np.clip(corrected * (1.0 - star_core[:, :, None] * 0.22) + white_star * star_core[:, :, None] * 0.22, 0.0, 1.0)


def _suppress_galaxy_background_smudges(
    image: np.ndarray,
    star_mask: np.ndarray,
    galaxy_signal: np.ndarray,
) -> np.ndarray:
    luminance = _luminance(image)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    chroma = np.maximum.reduce([blue, green, red]) - np.minimum.reduce([blue, green, red])

    star_core = np.clip(star_mask * 1.6, 0.0, 1.0)
    star_halo = cv2.GaussianBlur(star_core.astype(np.float32), (0, 0), 2.8)
    large_halo = cv2.GaussianBlur(star_core.astype(np.float32), (0, 0), 8.0)
    ring_artifact = np.clip(large_halo - star_halo * 0.72, 0.0, 1.0)

    bright_smudge = _smoothstep(
        np.percentile(luminance, 78),
        np.percentile(luminance, 98.5),
        cv2.GaussianBlur(luminance.astype(np.float32), (0, 0), 2.2),
    )
    color_smudge = _smoothstep(
        np.percentile(chroma, 76),
        np.percentile(chroma, 98.0),
        cv2.GaussianBlur(chroma.astype(np.float32), (0, 0), 2.0),
    )
    artifact_mask = np.maximum(ring_artifact, bright_smudge * color_smudge)
    artifact_mask *= (1.0 - np.clip(galaxy_signal * 1.6, 0.0, 1.0))
    artifact_mask *= (1.0 - star_core * 0.98)
    artifact_mask = cv2.GaussianBlur(np.clip(artifact_mask, 0.0, 1.0).astype(np.float32), (0, 0), 2.0)
    artifact_mask = np.clip(artifact_mask * 0.82, 0.0, 0.82)

    background = cv2.medianBlur(np.clip(image * 255.0, 0, 255).astype(np.uint8), 11).astype(np.float32) / 255.0
    background = cv2.bilateralFilter(
        np.clip(background * 255.0, 0, 255).astype(np.uint8),
        d=0,
        sigmaColor=12,
        sigmaSpace=26,
    ).astype(np.float32) / 255.0
    return np.clip(image * (1.0 - artifact_mask[:, :, None]) + background * artifact_mask[:, :, None], 0.0, 1.0)


def _polish_galaxy_background_grain(
    image: np.ndarray,
    star_mask: np.ndarray,
    galaxy_signal: np.ndarray,
) -> np.ndarray:
    luminance = _luminance(image)
    low_signal = _smoothstep(
        np.percentile(luminance, 4),
        np.percentile(luminance, 82),
        luminance,
    )
    background_mask = (1.0 - np.clip(galaxy_signal * 2.15, 0.0, 1.0)) * (1.0 - np.clip(star_mask * 1.4, 0.0, 1.0))
    background_mask *= np.clip(1.0 - low_signal * 0.42, 0.35, 1.0)
    background_mask = cv2.GaussianBlur(background_mask.astype(np.float32), (0, 0), 5.5)
    background_mask = np.clip(background_mask * 0.62, 0.0, 0.62)

    source = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    nlm = cv2.fastNlMeansDenoisingColored(
        source,
        None,
        h=9,
        hColor=11,
        templateWindowSize=7,
        searchWindowSize=25,
    ).astype(np.float32) / 255.0
    bilateral = cv2.bilateralFilter(source, d=0, sigmaColor=18, sigmaSpace=34).astype(np.float32) / 255.0
    smooth = np.clip(nlm * 0.6 + bilateral * 0.4, 0.0, 1.0)

    smooth_luminance = _luminance(smooth)
    ratio = np.clip(luminance / np.maximum(smooth_luminance, 1e-5), 0.82, 1.18)
    smooth = np.clip(smooth * ratio[:, :, None], 0.0, 1.0)

    neutral = np.repeat(_luminance(smooth)[:, :, None], 3, axis=2)
    chroma_mix = np.clip(background_mask * 0.34, 0.0, 0.34)
    smooth = np.clip(smooth * (1.0 - chroma_mix[:, :, None]) + neutral * chroma_mix[:, :, None], 0.0, 1.0)
    return np.clip(image * (1.0 - background_mask[:, :, None]) + smooth * background_mask[:, :, None], 0.0, 1.0)


def _genai_like_background_cleanup(
    image: np.ndarray,
    star_mask: np.ndarray,
    galaxy_signal: np.ndarray,
) -> np.ndarray:
    """Approximate masked cosmetic cleanup with deterministic OpenCV operations."""

    luminance = _luminance(image).astype(np.float32)
    galaxy_protect = _primary_object_protection(luminance)
    broad_galaxy = cv2.GaussianBlur(np.clip(galaxy_signal, 0.0, 1.0).astype(np.float32), (0, 0), 13.0)
    galaxy_protect = np.maximum(galaxy_protect, np.clip(broad_galaxy * 0.72, 0.0, 1.0))
    star_protect = cv2.GaussianBlur(np.clip(star_mask * 0.22, 0.0, 1.0).astype(np.float32), (0, 0), 1.2)
    protected = np.clip(np.maximum(galaxy_protect, star_protect), 0.0, 1.0)

    sky_gate = np.clip(1.0 - protected, 0.0, 1.0)
    sky_gate *= np.clip(1.0 - _smoothstep(np.percentile(luminance, 56), np.percentile(luminance, 92), luminance) * 0.45, 0.4, 1.0)
    sky_gate = cv2.GaussianBlur(sky_gate.astype(np.float32), (0, 0), 5.0)

    source = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    median_large = cv2.medianBlur(source, 21).astype(np.float32) / 255.0
    bilateral_large = cv2.bilateralFilter(source, d=0, sigmaColor=24, sigmaSpace=58).astype(np.float32) / 255.0
    nlm_strong = cv2.fastNlMeansDenoisingColored(
        source,
        None,
        h=13,
        hColor=15,
        templateWindowSize=7,
        searchWindowSize=31,
    ).astype(np.float32) / 255.0
    clean_sky_texture = np.clip(median_large * 0.40 + bilateral_large * 0.34 + nlm_strong * 0.26, 0.0, 1.0)
    clean_luminance = _luminance(clean_sky_texture)
    sky_pixels = sky_gate > 0.45
    if np.count_nonzero(sky_pixels) > 512:
        sky_level = float(np.percentile(clean_luminance[sky_pixels], 28))
    else:
        sky_level = float(np.percentile(clean_luminance, 18))
    clean_sky = np.full_like(image, np.clip(sky_level * 0.16 + 0.002, 0.002, 0.009), dtype=np.float32)

    local_background = cv2.GaussianBlur(luminance, (0, 0), 7.0)
    high_frequency = np.abs(luminance - local_background)
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    chroma = np.maximum.reduce([blue, green, red]) - np.minimum.reduce([blue, green, red])
    grain_mask = _smoothstep(
        np.percentile(high_frequency, 72),
        np.percentile(high_frequency, 98.8),
        high_frequency,
    )
    chroma_noise_mask = _smoothstep(
        np.percentile(chroma, 70),
        np.percentile(chroma, 98.0),
        cv2.GaussianBlur(chroma.astype(np.float32), (0, 0), 1.3),
    )
    blob_mask = _smoothstep(
        np.percentile(local_background, 72),
        np.percentile(local_background, 98.2),
        local_background,
    )
    artifact_mask = np.maximum(grain_mask * 0.42 + chroma_noise_mask * 0.36, blob_mask * 0.55)
    artifact_mask = np.clip(artifact_mask * sky_gate * 1.45, 0.0, 0.96)
    broad_cleanup = np.clip(sky_gate, 0.0, 1.0)
    cleanup_mask = np.maximum(broad_cleanup, artifact_mask)
    cleanup_mask = cv2.GaussianBlur(cleanup_mask.astype(np.float32), (0, 0), 3.0)
    cleanup_mask = np.clip(cleanup_mask, 0.0, 1.0)

    cleaned = np.clip(image * (1.0 - cleanup_mask[:, :, None]) + clean_sky * cleanup_mask[:, :, None], 0.0, 1.0)

    cleaned_luminance = _luminance(cleaned)
    sky_shadow = np.clip((0.055 - cleaned_luminance) / 0.055, 0.0, 1.0)
    sky_darkener = np.clip(sky_gate * (0.26 + sky_shadow * 0.22), 0.0, 0.42)
    return np.clip(cleaned * (1.0 - sky_darkener[:, :, None]), 0.0, 1.0)


def _primary_object_protection(luminance: np.ndarray) -> np.ndarray:
    smoothed = cv2.GaussianBlur(luminance.astype(np.float32), (0, 0), 10.0)
    threshold = np.percentile(smoothed, 93.0)
    binary = (smoothed > threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return cv2.GaussianBlur(
            _smoothstep(np.percentile(smoothed, 82), np.percentile(smoothed, 99.6), smoothed).astype(np.float32),
            (0, 0),
            8.0,
        )

    largest_label = max(range(1, count), key=lambda label: stats[label, cv2.CC_STAT_AREA])
    mask = (labels == largest_label).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((31, 31), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 18.0)
    soft_signal = _smoothstep(np.percentile(smoothed, 96), np.percentile(smoothed, 99.7), smoothed)
    mask = np.maximum(mask, soft_signal * 0.04)
    return np.clip(mask, 0.0, 1.0)


def _clean_primary_galaxy_object(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image).astype(np.float32)
    object_mask = cv2.GaussianBlur(_primary_object_protection(luminance).astype(np.float32), (0, 0), 5.0)
    if float(np.max(object_mask)) <= 0.0:
        return image

    source = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    nlm = cv2.fastNlMeansDenoisingColored(
        source,
        None,
        h=5,
        hColor=7,
        templateWindowSize=7,
        searchWindowSize=19,
    ).astype(np.float32) / 255.0
    bilateral = cv2.bilateralFilter(source, d=0, sigmaColor=14, sigmaSpace=18).astype(np.float32) / 255.0
    smooth = np.clip(nlm * 0.35 + bilateral * 0.35 + image * 0.30, 0.0, 1.0)

    smooth_luminance = _luminance(smooth)
    original_large = cv2.GaussianBlur(luminance, (0, 0), 7.0)
    smooth_large = cv2.GaussianBlur(smooth_luminance, (0, 0), 7.0)
    ratio = np.clip(original_large / np.maximum(smooth_large, 1e-5), 0.65, 1.45)
    smooth = np.clip(smooth * ratio[:, :, None], 0.0, 1.0)

    denoised_luminance = cv2.fastNlMeansDenoising(
        np.clip(luminance * 255.0, 0, 255).astype(np.uint8),
        None,
        h=6,
        templateWindowSize=7,
        searchWindowSize=19,
    ).astype(np.float32) / 255.0
    middle_scale = cv2.GaussianBlur(denoised_luminance, (0, 0), 2.4)
    large_scale = cv2.GaussianBlur(denoised_luminance, (0, 0), 15.0)
    arms = cv2.GaussianBlur(middle_scale - large_scale, (0, 0), 2.0)
    smooth_luminance = _luminance(smooth)
    enhanced_luminance = np.clip(smooth_luminance + arms * object_mask * 0.18, 0.0, 1.0)
    smooth = np.clip(smooth * (enhanced_luminance / np.maximum(smooth_luminance, 1e-5))[:, :, None], 0.0, 1.0)

    smooth_luminance = _luminance(smooth)
    object_luminance = luminance[object_mask > 0.18]
    if object_luminance.size > 128:
        core = _smoothstep(np.percentile(object_luminance, 72), np.percentile(object_luminance, 99.6), luminance)
    else:
        core = _smoothstep(np.percentile(luminance, 96.5), np.percentile(luminance, 99.9), luminance)
    core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 3.2)
    inner_disk = cv2.GaussianBlur(np.clip(object_mask - core * 0.25, 0.0, 1.0).astype(np.float32), (0, 0), 4.0)
    lifted_luminance = np.clip(
        smooth_luminance
        + inner_disk * np.power(np.clip(smooth_luminance, 0.0, 1.0), 0.78) * 0.045
        + core * np.power(np.clip(smooth_luminance, 0.0, 1.0), 0.62) * 0.13,
        0.0,
        1.0,
    )
    highlight_guard = _smoothstep(0.84, 0.98, smooth_luminance)
    lifted_luminance = smooth_luminance * highlight_guard + lifted_luminance * (1.0 - highlight_guard)
    smooth = np.clip(smooth * (lifted_luminance / np.maximum(smooth_luminance, 1e-5))[:, :, None], 0.0, 1.0)

    warmed = smooth.copy()
    warmed[:, :, 0] *= 0.92
    warmed[:, :, 1] *= 1.00
    warmed[:, :, 2] *= 1.08
    smooth = np.clip(smooth * (1.0 - object_mask[:, :, None] * 0.35) + warmed * object_mask[:, :, None] * 0.35, 0.0, 1.0)
    smooth_luminance = _luminance(smooth)
    core_white = _smoothstep(
        np.percentile(smooth_luminance[object_mask > 0.18], 78) if np.any(object_mask > 0.18) else np.percentile(smooth_luminance, 97),
        np.percentile(smooth_luminance[object_mask > 0.18], 99.7) if np.any(object_mask > 0.18) else np.percentile(smooth_luminance, 99.8),
        smooth_luminance,
    )
    core_white = cv2.GaussianBlur((core_white * object_mask).astype(np.float32), (0, 0), 2.5)
    neutral_core = np.repeat(smooth_luminance[:, :, None], 3, axis=2)
    smooth = np.clip(smooth * (1.0 - core_white[:, :, None] * 0.62) + neutral_core * core_white[:, :, None] * 0.62, 0.0, 1.0)

    mix = np.clip(object_mask * 0.34, 0.0, 0.52)
    return np.clip(image * (1.0 - mix[:, :, None]) + smooth * mix[:, :, None], 0.0, 1.0)


def _restore_galaxy_luminance_gradient(
    image: np.ndarray,
    original: np.ndarray,
    star_mask: np.ndarray,
) -> np.ndarray:
    luminance = _luminance(image).astype(np.float32)
    original_luminance = _luminance(original).astype(np.float32)
    object_mask = cv2.GaussianBlur(_primary_object_protection(luminance).astype(np.float32), (0, 0), 8.0)
    object_mask = np.clip(object_mask * (1.0 - cv2.GaussianBlur(star_mask.astype(np.float32), (0, 0), 1.2) * 0.7), 0.0, 1.0)
    object_pixels = object_mask > 0.18
    if np.count_nonzero(object_pixels) < 256:
        return image

    source_gradient = cv2.GaussianBlur(original_luminance, (0, 0), 5.5)
    source_low = float(np.percentile(source_gradient[object_pixels], 18))
    source_high = float(np.percentile(source_gradient[object_pixels], 99.85))
    if source_high <= source_low:
        return image

    source_tone = np.clip((source_gradient - source_low) / max(source_high - source_low, 1e-5), 0.0, 1.0)
    source_tone = np.arcsinh(source_tone * 2.2) / np.arcsinh(2.2)
    source_tone = np.power(source_tone, 1.24)

    current_low = float(np.percentile(luminance[object_pixels], 20))
    current_high = float(np.percentile(luminance[object_pixels], 99.65))
    target_luminance = current_low + source_tone * max(current_high - current_low, 1e-5) * 1.02
    target_luminance = np.clip(target_luminance, 0.0, 0.88)

    core = _smoothstep(
        np.percentile(source_tone[object_pixels], 76),
        np.percentile(source_tone[object_pixels], 99.5),
        source_tone,
    )
    core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 2.8)
    mix = np.clip(object_mask * (0.10 + core * 0.28), 0.0, 0.38)
    restored_luminance = luminance * (1.0 - mix) + target_luminance * mix
    highlight_guard = _smoothstep(0.82, 0.96, luminance)
    restored_luminance = restored_luminance * (1.0 - highlight_guard * 0.55) + luminance * (highlight_guard * 0.55)

    ratio = np.clip(restored_luminance / np.maximum(luminance, 1e-5), 0.72, 1.85)
    return np.clip(image * ratio[:, :, None], 0.0, 1.0)


def _finalize_enhanced_black_sky(
    composite: np.ndarray,
    galaxy_base: np.ndarray,
    synthetic_stars: np.ndarray,
) -> np.ndarray:
    object_protect = cv2.GaussianBlur(
        np.clip(_primary_object_protection(_luminance(galaxy_base)) * 1.45, 0.0, 1.0).astype(np.float32),
        (0, 0),
        7.0,
    )
    base_luminance = cv2.GaussianBlur(_luminance(galaxy_base).astype(np.float32), (0, 0), 14.0)
    faint_halo_protect = _smoothstep(
        np.percentile(base_luminance, 68),
        np.percentile(base_luminance, 98.8),
        base_luminance,
    )
    faint_halo_protect = cv2.GaussianBlur(faint_halo_protect.astype(np.float32), (0, 0), 12.0)
    object_protect = np.maximum(object_protect, np.clip(faint_halo_protect * 0.52, 0.0, 1.0))
    star_luminance = _luminance(synthetic_stars)
    star_pixels = star_luminance[star_luminance > 0.0003]
    if star_pixels.size > 0:
        star_core = _smoothstep(
            max(float(np.percentile(star_pixels, 10)), 0.0006),
            max(float(np.percentile(star_pixels, 92)), 0.0012),
            star_luminance,
        )
    else:
        star_core = np.zeros_like(star_luminance, dtype=np.float32)
    star_core = cv2.dilate((star_core > 0.08).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
    star_protect = cv2.GaussianBlur(star_core, (0, 0), 1.0)
    protected = np.clip(np.maximum(object_protect, star_protect), 0.0, 1.0)
    sky_mask = cv2.GaussianBlur(np.clip(1.0 - protected, 0.0, 1.0).astype(np.float32), (0, 0), 1.8)
    sky_mask = np.clip(sky_mask * 0.98, 0.0, 0.98)
    dark_sky = np.full_like(composite, 0.0025, dtype=np.float32)
    return np.clip(composite * (1.0 - sky_mask[:, :, None]) + dark_sky * sky_mask[:, :, None], 0.0, 1.0)


def _minimize_restored_star_halos(
    image: np.ndarray,
    original: np.ndarray,
    starless: np.ndarray,
    galaxy_base: np.ndarray,
) -> np.ndarray:
    """Neutralize colored halo rings after galaxy star restoration."""

    residual = _extract_star_layer(original, starless)
    residual_luminance = _luminance(residual).astype(np.float32)
    image_luminance = _luminance(image).astype(np.float32)
    star_mask = _build_star_mask(residual)
    galaxy_signal = _galaxy_signal_from_original(galaxy_base, star_mask)
    galaxy_protect = cv2.GaussianBlur(np.clip(galaxy_signal * 2.4, 0.0, 1.0).astype(np.float32), (0, 0), 8.0)

    threshold = max(float(np.percentile(residual_luminance, 98.7)), 0.0035)
    centers = (residual_luminance > threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(centers, 8)
    height, width = residual_luminance.shape
    core_mask = np.zeros((height, width), dtype=np.float32)
    ring_mask = np.zeros((height, width), dtype=np.float32)

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 1 or area > 260:
            continue
        cx, cy = centroids[label]
        peak = float(np.max(residual_luminance[labels == label]))
        measured_radius = float(np.sqrt(area / np.pi))
        core_radius = float(np.clip(measured_radius * (1.0 + peak * 2.2), 0.9, 3.2))
        inner_radius = float(np.clip(measured_radius * (2.1 + peak * 5.0), 2.0, 6.5))
        outer_radius = float(np.clip(measured_radius * (4.2 + peak * 11.0), 3.5, 12.0))
        x0 = max(0, int(cx - outer_radius * 2.4))
        x1 = min(width, int(cx + outer_radius * 2.4 + 1))
        y0 = max(0, int(cy - outer_radius * 2.4))
        y1 = min(height, int(cy + outer_radius * 2.4 + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance_squared = (xx - cx) ** 2 + (yy - cy) ** 2
        core = np.exp(-distance_squared / (2.0 * core_radius * core_radius))
        inner = np.exp(-distance_squared / (2.0 * inner_radius * inner_radius))
        outer = np.exp(-distance_squared / (2.0 * outer_radius * outer_radius))
        annulus = np.clip(outer - inner * 0.70 - core * 0.18, 0.0, 1.0)
        local_protect = np.clip(1.0 - galaxy_protect[y0:y1, x0:x1] * 0.86, 0.10, 1.0)
        core_mask[y0:y1, x0:x1] = np.maximum(core_mask[y0:y1, x0:x1], core.astype(np.float32))
        ring_mask[y0:y1, x0:x1] = np.maximum(
            ring_mask[y0:y1, x0:x1],
            (annulus * local_protect).astype(np.float32),
        )

    core_mask = cv2.GaussianBlur(np.clip(core_mask, 0.0, 1.0), (0, 0), 0.45)
    ring_mask = cv2.GaussianBlur(np.clip(ring_mask, 0.0, 1.0), (0, 0), 1.05)

    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]
    chroma = np.maximum.reduce([blue, green, red]) - np.minimum.reduce([blue, green, red])
    broad_luminance = cv2.GaussianBlur(image_luminance, (0, 0), 1.7)
    visible = _smoothstep(
        max(float(np.percentile(broad_luminance, 61)), 0.003),
        max(float(np.percentile(broad_luminance, 96)), 0.012),
        broad_luminance,
    )
    colorful = _smoothstep(
        max(float(np.percentile(chroma, 64)), 0.004),
        max(float(np.percentile(chroma, 97)), 0.018),
        cv2.GaussianBlur(chroma.astype(np.float32), (0, 0), 1.3),
    )
    halo_mask = np.clip(
        ring_mask * (0.28 + visible * 0.48 + colorful * 0.70) * (1.0 - core_mask * 0.995),
        0.0,
        1.0,
    )
    halo_mask = cv2.GaussianBlur(halo_mask.astype(np.float32), (0, 0), 0.75)
    if float(np.max(halo_mask)) <= 0.0:
        return image

    source = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    median7 = cv2.medianBlur(source, 7).astype(np.float32) / 255.0
    median11 = cv2.medianBlur(source, 11).astype(np.float32) / 255.0
    bilateral = cv2.bilateralFilter(source, d=0, sigmaColor=7, sigmaSpace=18).astype(np.float32) / 255.0
    local_sky = np.clip(median7 * 0.34 + median11 * 0.34 + bilateral * 0.32, 0.0, 1.0)
    local_luminance = _luminance(local_sky)
    local_sky = np.clip(
        local_sky * np.clip(image_luminance / np.maximum(local_luminance, 1e-5), 0.92, 1.06)[:, :, None],
        0.0,
        1.0,
    )

    neutral = np.repeat(image_luminance[:, :, None], 3, axis=2)
    replace_mask = np.clip(halo_mask * 0.20, 0.0, 0.48)
    neutral_mask = np.clip(halo_mask * 0.72, 0.0, 0.76)
    cleaned = np.clip(image * (1.0 - neutral_mask[:, :, None]) + neutral * neutral_mask[:, :, None], 0.0, 1.0)
    cleaned = np.clip(cleaned * (1.0 - replace_mask[:, :, None]) + local_sky * replace_mask[:, :, None], 0.0, 1.0)

    cleaned_luminance = _luminance(cleaned)
    cleaned_neutral = np.repeat(cleaned_luminance[:, :, None], 3, axis=2)
    fringe_mask = np.clip(halo_mask * 0.24 * (1.0 - core_mask * 0.96), 0.0, 0.34)
    return np.clip(
        cleaned * (1.0 - fringe_mask[:, :, None]) + cleaned_neutral * fringe_mask[:, :, None],
        0.0,
        1.0,
    )


def _neutralize_galaxy_background_color(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    galaxy_signal = _galaxy_signal_from_original(image, star_mask)
    background_mask = (
        (luminance < np.percentile(luminance, 62))
        & (galaxy_signal < 0.16)
        & (star_mask < 0.08)
    )
    if np.count_nonzero(background_mask) < 512:
        return image

    background = np.median(image[background_mask], axis=0)
    neutral = float(np.mean(background))
    gains = np.clip(neutral / np.maximum(background, 1e-4), 0.72, 1.28)
    corrected = np.clip(image * gains.reshape(1, 1, 3), 0.0, 1.0)

    background_mix = np.clip((1.0 - galaxy_signal) * (1.0 - star_mask) * 0.86, 0.0, 0.86)
    galaxy_mix = np.clip(galaxy_signal * 0.22, 0.0, 0.22)
    mix = np.maximum(background_mix, galaxy_mix)
    return np.clip(image * (1.0 - mix[:, :, None]) + corrected * mix[:, :, None], 0.0, 1.0)


def _galaxy_signal_from_original(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    luminance = _luminance(image).astype(np.float32)
    star_suppressed = luminance * (1.0 - star_mask * 0.85) + np.median(luminance) * star_mask * 0.85
    smooth = cv2.GaussianBlur(star_suppressed, (0, 0), 9.0)
    low = np.percentile(smooth, 72)
    high = np.percentile(smooth, 99.4)
    if high <= low:
        return np.zeros_like(luminance)
    signal = np.clip((smooth - low) / (high - low), 0.0, 1.0)
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 5.0)
    return np.clip(signal, 0.0, 1.0)


def _soft_match_luminance(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_luminance = _luminance(source)
    reference_luminance = _luminance(reference)
    source_low, source_high = np.percentile(source_luminance, [3, 99.5])
    ref_low, ref_high = np.percentile(reference_luminance, [3, 99.5])
    if source_high <= source_low or ref_high <= ref_low:
        return source
    matched = (source - source_low) / max(source_high - source_low, 1e-4)
    matched = matched * (ref_high - ref_low) + ref_low
    return np.clip(matched, 0.0, 1.0)


def _reduce_galaxy_background_chroma_noise(
    image: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(np.clip(image * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    a_smooth = cv2.bilateralFilter(a_channel, d=7, sigmaColor=18, sigmaSpace=11)
    b_smooth = cv2.bilateralFilter(b_channel, d=7, sigmaColor=18, sigmaSpace=11)
    filtered = cv2.cvtColor(cv2.merge([l_channel, a_smooth, b_smooth]), cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    mix = np.clip((1.0 - protected)[:, :, None] * 0.5, 0.0, 0.5)
    return np.clip(image * (1.0 - mix) + filtered * mix, 0.0, 1.0)


def _denoise_galaxy_scene_without_repainting(
    image: np.ndarray,
    protected: np.ndarray,
) -> np.ndarray:
    source = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(
        source,
        None,
        h=5.0,
        hColor=8.0,
        templateWindowSize=7,
        searchWindowSize=21,
    ).astype(np.float32) / 255.0
    mix = np.clip((1.0 - protected)[:, :, None] * 0.68 + protected[:, :, None] * 0.18, 0.0, 0.68)
    return np.clip(image * (1.0 - mix) + denoised * mix, 0.0, 1.0)


def _finish_galaxy_from_original(
    image: np.ndarray,
    galaxy_signal: np.ndarray,
    star_mask: np.ndarray,
) -> np.ndarray:
    luminance = _luminance(image)
    background_mask = (galaxy_signal < 0.18) & (star_mask < 0.12)
    if np.count_nonzero(background_mask) < 512:
        background_mask = luminance < np.percentile(luminance, 55)

    black = np.percentile(luminance[background_mask], 50)
    white = np.percentile(luminance, 99.82)
    scaled = np.clip((image - black) / max(white - black, 1e-4), 0.0, 1.0)
    scaled = np.power(scaled, 1.12)

    luminance = _luminance(scaled)
    highlight = luminance > 0.72
    if np.any(highlight):
        compressed = luminance.copy()
        compressed[highlight] = 0.72 + np.tanh((luminance[highlight] - 0.72) / 0.2) * 0.2
        scaled = np.clip(scaled * (compressed / np.maximum(luminance, 1e-5))[:, :, None], 0.0, 1.0)

    hsv = cv2.cvtColor(np.clip(scaled * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.03 + galaxy_signal * 0.14), 0, 255)
    toned = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    shadow = np.clip((0.24 - _luminance(toned)) / 0.24, 0.0, 1.0)
    background_shadow = shadow * (1.0 - np.clip(galaxy_signal * 1.5, 0.0, 1.0)) * (1.0 - star_mask * 0.8)
    toned = np.clip(toned * (1.0 - background_shadow[:, :, None] * 0.28), 0.0, 1.0)
    return toned


def _recover_galaxy_core_and_arms(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    galaxy_mask = np.clip((luminance - np.percentile(luminance, 78)) / 0.35, 0.0, 1.0)
    galaxy_mask = cv2.GaussianBlur(galaxy_mask.astype(np.float32), (0, 0), 9.0)

    blurred = cv2.GaussianBlur(image, (0, 0), 5.0)
    detail = np.clip(image - blurred + 0.5, 0.0, 1.0) - 0.5
    arms = np.clip(image + detail * galaxy_mask[:, :, None] * 1.6, 0.0, 1.0)

    lum = _luminance(arms)
    core = lum > np.percentile(lum, 99.35)
    compressed_lum = lum.copy()
    shoulder = np.percentile(lum, 99.35)
    compressed_lum[core] = shoulder + np.tanh((lum[core] - shoulder) / 0.22) * 0.12
    ratio = compressed_lum / np.maximum(lum, 1e-5)
    recovered = np.clip(arms * ratio[:, :, None], 0.0, 1.0)
    return np.clip(image * (1.0 - galaxy_mask[:, :, None] * 0.85) + recovered * galaxy_mask[:, :, None] * 0.85, 0.0, 1.0)


def _finish_galaxy_scene(
    image: np.ndarray,
    star_mask: np.ndarray,
    preserve_color_calibration: bool = False,
) -> np.ndarray:
    luminance = _luminance(image)
    black = np.percentile(luminance, 15.0)
    white = np.percentile(luminance, 99.94)
    balanced = np.clip((image - black) / max(white - black, 1e-4), 0.0, 1.0)
    balanced = np.power(balanced, 1.18)

    luminance = _luminance(balanced)
    galaxy_signal = cv2.GaussianBlur(
        np.clip((luminance - np.percentile(luminance, 84)) / 0.3, 0.0, 1.0).astype(np.float32),
        (0, 0),
        9.0,
    )
    protected = np.clip(np.maximum(galaxy_signal, star_mask), 0.0, 1.0)
    background_mask = (luminance < np.percentile(luminance, 70)) & (protected < 0.18)
    if np.count_nonzero(background_mask) > 512:
        background = np.median(balanced[background_mask], axis=0)
        neutral = float(np.mean(background))
        gains = np.clip(neutral / np.maximum(background, 1e-4), 0.82, 1.18)
        corrected = np.clip(balanced * gains.reshape(1, 1, 3), 0.0, 1.0)
        shadow_mix = np.clip((0.46 - luminance) / 0.46, 0.0, 1.0) * (1.0 - protected * 0.85)
        balanced = balanced * (1.0 - shadow_mix[:, :, None] * 0.75) + corrected * shadow_mix[:, :, None] * 0.75

    background_smooth = cv2.bilateralFilter(
        np.clip(balanced * 255.0, 0, 255).astype(np.uint8),
        d=11,
        sigmaColor=26,
        sigmaSpace=21,
    ).astype(np.float32) / 255.0
    smooth_mix = np.clip((1.0 - protected)[:, :, None] * 0.7, 0.0, 0.7)
    balanced = np.clip(balanced * (1.0 - smooth_mix) + background_smooth * smooth_mix, 0.0, 1.0)

    luminance = _luminance(balanced)
    highlight = luminance > 0.4
    if np.any(highlight):
        compressed = luminance.copy()
        compressed[highlight] = 0.4 + np.tanh((luminance[highlight] - 0.4) / 0.2) * 0.28
        ratio = compressed / np.maximum(luminance, 1e-5)
        balanced = np.clip(balanced * ratio[:, :, None], 0.0, 1.0)

    luminance = _luminance(balanced)
    bright_galaxy = cv2.GaussianBlur(
        np.clip((luminance - np.percentile(luminance, 97.8)) / 0.22, 0.0, 1.0).astype(np.float32),
        (0, 0),
        7.0,
    )
    if np.any(bright_galaxy > 0.01):
        glow_tamed = np.power(np.clip(balanced, 0.0, 1.0), 1.22)
        balanced = np.clip(
            balanced * (1.0 - bright_galaxy[:, :, None] * 0.48)
            + glow_tamed * bright_galaxy[:, :, None] * 0.48,
            0.0,
            1.0,
        )

    luminance = _luminance(balanced)
    star_soften = np.clip(star_mask[:, :, None] * 0.42, 0.0, 0.42)
    star_tamed = cv2.GaussianBlur(balanced, (0, 0), 0.8)
    balanced = np.clip(balanced * (1.0 - star_soften) + star_tamed * star_soften, 0.0, 1.0)

    galaxy_signal = cv2.GaussianBlur(np.clip((luminance - np.percentile(luminance, 84)) / 0.3, 0.0, 1.0).astype(np.float32), (0, 0), 5.0)
    if preserve_color_calibration:
        return np.clip(balanced, 0.0, 1.0)

    warmed = balanced.copy()
    warmed[:, :, 2] *= 1.09
    warmed[:, :, 1] *= 0.98
    warmed[:, :, 0] *= 0.9
    mix = np.clip(galaxy_signal[:, :, None] * 0.5, 0.0, 0.5)
    return np.clip(balanced * (1.0 - mix) + warmed * mix, 0.0, 1.0)


def _set_final_black_point(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    emission = _emission_mask(image)
    black = np.percentile(luminance[emission < 0.18], 8.5) if np.any(emission < 0.18) else np.percentile(luminance, 8.5)
    white = np.percentile(luminance, 99.85)
    scaled = np.clip((image - black) / max(white - black, 1e-4), 0.0, 1.0)
    luminance_scaled = _luminance(scaled)
    shadows = np.clip(1.0 - luminance_scaled / 0.38, 0.0, 1.0)
    emission_scaled = _emission_mask(scaled)
    shadow_strength = shadows * (1.0 - np.clip(emission_scaled * 1.25, 0.0, 0.82))
    scaled = scaled * (1.0 - shadow_strength[:, :, None] * 0.15)
    return np.clip(scaled, 0.0, 1.0)


def _blend_detail_aware_denoise(starless: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    detail = _detail_mask(starless)
    smooth_strength = np.clip(1.0 - detail[:, :, None] * 0.82, 0.28, 0.88)
    return np.clip(starless * (1.0 - smooth_strength) + denoised * smooth_strength, 0.0, 1.0)


def _extract_star_layer(original: np.ndarray, starless: np.ndarray) -> np.ndarray:
    residual = np.clip(original - starless, 0.0, 1.0)
    residual = np.where(residual > 0.006, residual, 0.0)
    return residual


def _build_star_mask(stars: np.ndarray) -> np.ndarray:
    luminance = _luminance(stars)
    threshold = max(float(np.percentile(luminance, 99.1)), 0.018)
    mask = (luminance > threshold).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.2)
    return mask.astype(np.float32) / 255.0


def _suppress_starless_halos(nebula: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(nebula, (0, 0), 6.0)
    mask = np.clip(star_mask[:, :, None] * 0.85, 0.0, 0.85)
    return np.clip(nebula * (1.0 - mask) + blurred * mask, 0.0, 1.0)


def _restore_stars(nebula: np.ndarray, stars: np.ndarray) -> np.ndarray:
    star_display = _hdr_stretch(stars)
    star_lum = _luminance(star_display)
    star_scale = np.clip(np.interp(star_lum, [0.0, 0.25, 1.0], [0.0, 0.42, 0.68]), 0.0, 1.0)
    restored = 1.0 - (1.0 - nebula) * (1.0 - star_display * star_scale[:, :, None])
    return np.clip(restored, 0.0, 1.0)


def _finish_nebula_dark_sky(image: np.ndarray, star_mask: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    emission = _emission_mask(image)
    broad_luminance = cv2.GaussianBlur(luminance.astype(np.float32), (0, 0), 10.0)
    nebula_signal = np.maximum(
        np.clip(emission * 1.35, 0.0, 1.0),
        _smoothstep(
            np.percentile(broad_luminance, 68),
            np.percentile(broad_luminance, 98.8),
            broad_luminance,
        ),
    )
    star_protect = cv2.GaussianBlur(np.clip(star_mask, 0.0, 1.0).astype(np.float32), (0, 0), 1.4)
    sky_pixels = (nebula_signal < 0.18) & (star_protect < 0.08) & (luminance < np.percentile(luminance, 86))
    if np.count_nonzero(sky_pixels) < 512:
        return image

    black = float(np.percentile(luminance[sky_pixels], 62))
    dark_scaled = np.clip((image - black * 0.72) / max(1.0 - black * 0.72, 1e-4), 0.0, 1.0)
    dark_scaled *= 0.82

    denoised = cv2.bilateralFilter(
        np.clip(dark_scaled * 255.0, 0, 255).astype(np.uint8),
        d=0,
        sigmaColor=18,
        sigmaSpace=28,
    ).astype(np.float32) / 255.0
    denoised = cv2.fastNlMeansDenoisingColored(
        np.clip(denoised * 255.0, 0, 255).astype(np.uint8),
        None,
        h=5,
        hColor=7,
        templateWindowSize=7,
        searchWindowSize=21,
    ).astype(np.float32) / 255.0

    low_luminance = 1.0 - _smoothstep(np.percentile(luminance, 58), np.percentile(luminance, 92), luminance)
    sky_mix = np.clip(low_luminance * (1.0 - nebula_signal) * (1.0 - star_protect), 0.0, 1.0)
    sky_mix = cv2.GaussianBlur(sky_mix.astype(np.float32), (0, 0), 2.8)
    sky_mix = np.clip(sky_mix * 0.68, 0.0, 0.68)
    darkened = np.clip(dark_scaled * 0.55 + denoised * 0.45, 0.0, 1.0)
    return np.clip(image * (1.0 - sky_mix[:, :, None]) + darkened * sky_mix[:, :, None], 0.0, 1.0)


def _cosmetic_cleanup(image: np.ndarray) -> np.ndarray:
    median = cv2.medianBlur(np.clip(image * 255.0, 0, 255).astype(np.uint8), 3)
    cleaned = cv2.addWeighted(np.clip(image * 255.0, 0, 255).astype(np.uint8), 0.86, median, 0.14, 0)
    return cleaned.astype(np.float32) / 255.0


def _auto_crop_dark_edges(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image)
    threshold = max(float(np.percentile(luminance, 1.5)) * 1.15, 0.01)
    rows = np.where(np.median(luminance, axis=1) > threshold)[0]
    cols = np.where(np.median(luminance, axis=0) > threshold)[0]
    if rows.size == 0 or cols.size == 0:
        return image
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    max_y_crop = image.shape[0] // 12
    max_x_crop = image.shape[1] // 12
    top = min(top, max_y_crop)
    left = min(left, max_x_crop)
    bottom = max(bottom, image.shape[0] - max_y_crop)
    right = max(right, image.shape[1] - max_x_crop)
    return image[top:bottom, left:right]


def _luminance(image: np.ndarray) -> np.ndarray:
    return image[:, :, 2] * 0.2126 + image[:, :, 1] * 0.7152 + image[:, :, 0] * 0.0722


def _is_probable_galaxy(image: np.ndarray) -> bool:
    luminance = _luminance(image)
    threshold = np.percentile(luminance, 98.4)
    bright = (luminance > threshold).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if num_labels <= 1:
        return False
    largest = max(stats[label, cv2.CC_STAT_AREA] for label in range(1, num_labels))
    image_area = image.shape[0] * image.shape[1]
    return largest / image_area > 0.004


def _detail_mask(image: np.ndarray) -> np.ndarray:
    luminance = _luminance(image).astype(np.float32)
    blurred = cv2.GaussianBlur(luminance, (0, 0), 3.0)
    detail = np.abs(luminance - blurred)
    high = np.percentile(detail, 98.8)
    if high <= 1e-5:
        return np.zeros_like(luminance)
    mask = np.clip(detail / high, 0.0, 1.0)
    return cv2.GaussianBlur(mask, (0, 0), 1.2)


def _emission_mask(image: np.ndarray) -> np.ndarray:
    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]
    luminance = _luminance(image)
    red_excess = red - (green * 0.62 + blue * 0.38)
    threshold = max(float(np.percentile(red_excess, 62)), 0.01)
    scale = max(float(np.percentile(red_excess, 96) - threshold), 0.02)
    mask = np.clip((red_excess - threshold) / scale, 0.0, 1.0)
    signal = np.clip((luminance - np.percentile(luminance, 18)) / 0.32, 0.0, 1.0)
    mask *= signal
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
