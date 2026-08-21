from argparse import Namespace
from ..pipeline import run_pipeline

def execute(args: Namespace, ctx: dict) -> None:
    run_pipeline(ctx, args.top_percent, args.ranking_file)
