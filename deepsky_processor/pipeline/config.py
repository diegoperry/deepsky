"""Configuration loading for the local DeepSky pipeline."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration read from environment variables."""

    verify_mode: str
    require_cuda: bool
    siril_cli: str
    starnet_path: Path | None
    starnet_args: list[str]
    scunet_model_path: Path | None
    scunet_device: str
    scunet_model_type: str


def _optional_path(env_name: str) -> Path | None:
    value = os.environ.get(env_name)
    if not value:
        return None
    return Path(value).expanduser()


def load_config() -> PipelineConfig:
    """Load pipeline configuration from environment variables."""

    verify_mode = os.environ.get("DEEPSKY_VERIFY_MODE", "local")
    return PipelineConfig(
        verify_mode=verify_mode,
        require_cuda=os.environ.get("DEEPSKY_REQUIRE_CUDA", "0") == "1",
        siril_cli=os.environ.get("SIRIL_CLI", "siril-cli"),
        starnet_path=_optional_path("STARNET_PATH"),
        starnet_args=shlex.split(os.environ.get("STARNET_ARGS", "{input} {output} 256")),
        scunet_model_path=_optional_path("SCUNET_MODEL_PATH"),
        scunet_device=os.environ.get("SCUNET_DEVICE", "cuda" if os.environ.get("DEEPSKY_REQUIRE_CUDA") == "1" else "cpu"),
        scunet_model_type=os.environ.get("SCUNET_MODEL_TYPE", "auto"),
    )
