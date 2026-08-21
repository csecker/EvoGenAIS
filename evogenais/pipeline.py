"""Orchestration for generation and ranking stages."""

from __future__ import annotations

import gzip
import math
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ConfigurationError, resolve_config_path


def ranking_path(ctx: dict[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        path = Path(override).expanduser().resolve()
    else:
        config = ctx["main_config"]
        default = Path(ctx["output_root"]) / "ranking_latest.csv.gz"
        path = resolve_config_path(config, "ranking_file", str(default))
    if not path.is_file() and ctx["iteration"] > 1:
        raise FileNotFoundError(f"Ranking file not found: {path}")
    return path


def create_seed_collection(
    ctx: dict[str, Any],
    top_percent: float,
    ranking_file: str | Path | None = None,
) -> Path:
    if not 0 < top_percent <= 100:
        raise ConfigurationError("top-percent must be greater than 0 and at most 100")
    source = ranking_path(ctx, ranking_file)
    frame = pd.read_csv(source, compression="infer")
    column = str(ctx["main_config"].get("seed_smiles_column", "attr_smi_orig"))
    if column not in frame.columns:
        raise ConfigurationError(f"Ranking file does not contain required column {column!r}: {source}")
    smiles = frame[column].dropna().astype(str).str.strip()
    smiles = smiles[smiles.ne("")].drop_duplicates(keep="first")
    if smiles.empty:
        raise ConfigurationError(f"No usable SMILES found in column {column!r}: {source}")
    count = max(1, math.ceil(len(smiles) * top_percent / 100.0))
    destination = Path(ctx["temp_dir"].name) / "ranking-seed.smi.gz"
    with gzip.open(destination, "wt", encoding="utf-8", newline="\n") as handle:
        for value in smiles.iloc[:count]:
            handle.write(f"{value}\n")
    return destination


def run_generation(
    ctx: dict[str, Any],
    top_percent: float,
    ranking_file: str | Path | None = None,
) -> Path:
    from .scripts.generate_molecules import process_iteration
    from .scripts.generate_ranking import process

    iteration = int(ctx["iteration"])

    # Skip ranking-based seed creation for 1st-iteration generation-only mode.
    if iteration == 1:
        process(ctx)

    else:
        collection = create_seed_collection(ctx, top_percent, ranking_file)
        iteration = int(ctx["iteration"])
        with tempfile.TemporaryDirectory(prefix="molecule-collection-") as collection_dir:
            class DirectoryRef:
                name = collection_dir

            task = {
                "iteration": iteration,
                "collection_temp_dir": ctx["temp_dir"],
                "collection_file": str(collection),
                "collection_file_tmp": None,
                "training_file": None,
                "training_folder": None,
                "validation_file": None,
                "validation_folder": None,
            }
            process_iteration(ctx, task)
        return collection


def run_ranking(ctx: dict[str, Any]) -> None:
    from .scripts.generate_ranking import process

    process(ctx)


def run_pipeline(
    ctx: dict[str, Any],
    top_percent: float,
    ranking_file: str | Path | None = None,
) -> None:
    run_generation(ctx, top_percent, ranking_file)
    run_ranking(ctx)
