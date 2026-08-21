"""Configuration loading and runtime context construction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when configuration is missing or malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON in {config_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError("The configuration root must be a JSON object.")
    main_config = data.get("main_config", data)
    if not isinstance(main_config, dict):
        raise ConfigurationError("'main_config' must be a JSON object.")
    main_config["_config_dir"] = str(config_path.parent)
    return main_config


def resolve_config_path(config: dict[str, Any], key: str, default: str | None = None) -> Path:
    raw = config.get(key, default)
    if raw is None:
        raise ConfigurationError(f"Missing required configuration key: {key}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def build_context(
    config_path: str | Path,
    iteration: int,
    temp_parent: str | Path | None = None,
) -> tuple[dict[str, Any], tempfile.TemporaryDirectory[str]]:
    if iteration < 1:
        raise ConfigurationError("Iteration must be 1 or greater.")

    main_config = load_config(config_path)

    configured_parent = (
        temp_parent
        if temp_parent is not None
        else main_config.get("collection_temp_dir")
    )

    parent = None

    if configured_parent is not None:
        parent_path = Path(configured_parent).expanduser()

        if not parent_path.is_absolute():
            config_dir = Path(config_path).expanduser().resolve().parent
            parent_path = config_dir / parent_path

        parent_path = parent_path.resolve()
        parent_path.mkdir(parents=True, exist_ok=True)
        parent = str(parent_path)

    temp_dir = tempfile.TemporaryDirectory(
        prefix="evogenais-",
        dir=parent,
    )

    context: dict[str, Any] = {
        "iteration": iteration,
        "main_config": main_config,
        "temp_dir": temp_dir,
        "output_root": str(
            resolve_config_path(
                main_config,
                "output_root",
                "output-files",
            )
        ),
    }

    return context, temp_dir
