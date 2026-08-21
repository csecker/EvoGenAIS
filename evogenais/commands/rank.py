from argparse import Namespace
from ..pipeline import run_ranking

def execute(args: Namespace, ctx: dict) -> None:
    del args
    run_ranking(ctx)
