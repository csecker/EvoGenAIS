"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .commands import generate as generate_command
from .commands import rank as rank_command
from .commands import run as run_command
from .config import ConfigurationError, build_context


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return number


def _percentage(value: str) -> float:
    number = float(value)
    if not 0 < number <= 100:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 100")
    return number


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", required=True, metavar="FILE")
    parser.add_argument("-i", "--iteration", required=True, type=_positive_integer)
    parser.add_argument("--temp-dir", metavar="DIR")


def _add_seed_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--top-percent",
        type=_percentage,
        default=None,
        help="Top percentage of ranking attr_smi_orig used as generation seeds; defaults to config.",
    )
    parser.add_argument(
        "--ranking-file",
        metavar="FILE",
        help="Override the ranking_latest.csv.gz path from config.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evogenais")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate molecules from the current ranking.")
    _add_common(generate); _add_seed_options(generate)
    generate.set_defaults(handler=generate_command.execute)
    rank = sub.add_parser("rank", help="Rank docking results.")
    _add_common(rank); rank.set_defaults(handler=rank_command.execute)
    run = sub.add_parser("run", help="Generate molecules, then rank docking results.")
    _add_common(run); _add_seed_options(run)
    run.set_defaults(handler=run_command.execute)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    runtime_temp = None
    try:
        ctx, runtime_temp = build_context(Path(args.config), args.iteration, args.temp_dir)
        if hasattr(args, "top_percent") and args.top_percent is None:
            args.top_percent = float(ctx["main_config"].get("top_percent", 10))
        args.handler(args, ctx)
        return 0
    except (ConfigurationError, FileNotFoundError, ValueError, KeyError) as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logging.error("Interrupted by user")
        return 130
    except Exception:
        logging.exception("Pipeline failed")
        return 1
    finally:
        if runtime_temp is not None:
            runtime_temp.cleanup()
