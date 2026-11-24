"""Command-line entry point for review, individual stages and isolated reproduction."""

import argparse

parser = argparse.ArgumentParser(description="CrisisMMD capstone reproduction")
parser.add_argument(
    "stage",
    choices=[
        "download",
        "prepare",
        "features",
        "train",
        "evaluate",
        "plots",
        "report",
        "verify",
        "reproduce",
    ],
)
parser.add_argument(
    "--output",
    help="New output folder for reproduce; default is a unique folder under runs/.",
)
parser.add_argument(
    "--reuse-data",
    action="store_true",
    help="Reuse existing raw downloads for reproduce; features and models are recomputed.",
)
args = parser.parse_args()
if args.stage != "reproduce" and (args.output or args.reuse_data):
    parser.error("--output and --reuse-data apply only to reproduce")
if args.stage == "reproduce":
    from .reproduce import run

    run(output=args.output, reuse_data=args.reuse_data)
elif args.stage in ["download", "prepare"]:
    from . import data

    getattr(data, args.stage)()
elif args.stage == "features":
    from .features import extract

    extract()
elif args.stage in ["train", "evaluate", "plots", "report", "verify"]:
    module = __import__("src." + args.stage, fromlist=["run"])
    module.run()
