from argparse import Namespace
from ..pipeline import run_generation

def execute(args: Namespace, ctx: dict) -> None:
    run_generation(ctx, args.top_percent, args.ranking_file)
