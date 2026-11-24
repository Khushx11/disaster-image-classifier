"""Run a fresh experiment in an isolated directory, preserving submitted evidence."""

from datetime import datetime
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from .data import ROOT

STAGES = ("download", "prepare", "features", "train", "evaluate", "plots", "verify")


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return str(destination)


def run(output=None, reuse_data=False):
    destination = (
        Path(output).expanduser().resolve()
        if output
        else ROOT / "runs" / datetime.now().strftime("reproduction_%Y%m%d_%H%M%S_%f")
    )
    if destination == ROOT or destination.exists():
        raise FileExistsError(
            "Choose a new output folder. Existing results are never overwritten."
        )
    if any(
        destination.is_relative_to(ROOT / name)
        for name in ["src", "data", "results", "work"]
    ):
        raise ValueError(
            "Use runs/ or a separate new folder outside source, data, results and work."
        )
    if ROOT.is_relative_to(destination):
        raise ValueError("The output folder cannot contain the source project.")
    if reuse_data and not (ROOT / "data/raw").is_dir():
        raise FileNotFoundError(
            "--reuse-data needs data/raw; omit it to download the public dataset."
        )
    destination.mkdir(parents=True)
    shutil.copytree(
        ROOT / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "protocol.json", destination / "protocol.json")
    (destination / "data").mkdir()
    (destination / "results").mkdir()
    if reuse_data:
        raw = ROOT / "data/raw"
        shutil.copytree(raw, destination / "data/raw", copy_function=_link_or_copy)
    print("Fresh reproduction folder:", destination, flush=True)
    completed = []
    for stage in STAGES:
        print("Running:", stage, flush=True)
        subprocess.run(
            [sys.executable, "-m", "src", stage], cwd=destination, check=True
        )
        completed.append(stage)
    (destination / "results/reproduction_execution.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "completed_stages": completed,
                "reused_raw_downloads": reuse_data,
                "reused_features_or_model_weights": False,
                "submitted_results_overwritten": False,
            },
            indent=2,
        )
    )
    print(
        "Reproduction complete. Original submission results are unchanged.", flush=True
    )
    return destination
