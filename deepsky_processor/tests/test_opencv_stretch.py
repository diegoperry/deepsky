from __future__ import annotations

import numpy as np

from deepsky_processor.pipeline.opencv_runner import _estimate_nebula_stretch


def test_nebula_stretch_estimator_is_stronger_for_clean_signal() -> None:
    y, x = np.mgrid[0:128, 0:128]
    center_signal = np.exp(-(((x - 64) / 22) ** 2 + ((y - 64) / 18) ** 2)).astype(np.float32)
    clean_luminance = np.clip(0.035 + center_signal * 0.42, 0.0, 1.0)
    noisy_luminance = np.clip(
        0.055
        + center_signal * 0.12
        + np.random.default_rng(7).normal(0.0, 0.026, clean_luminance.shape).astype(np.float32),
        0.0,
        1.0,
    )

    clean_emission = np.clip(center_signal, 0.0, 1.0)
    noisy_emission = np.clip(center_signal * 0.25, 0.0, 1.0)
    detail = np.zeros_like(clean_luminance, dtype=np.float32)
    sky_mask = center_signal < 0.08

    clean = _estimate_nebula_stretch(clean_luminance, clean_emission, detail, sky_mask)
    noisy = _estimate_nebula_stretch(noisy_luminance, noisy_emission, detail, sky_mask)

    assert clean["asinh_strength"] > noisy["asinh_strength"]
    assert clean["gamma"] < noisy["gamma"]
    assert clean["sky_discipline"] < noisy["sky_discipline"]
