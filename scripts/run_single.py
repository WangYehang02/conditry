#!/usr/bin/env python3
"""Train and evaluate FMGAD on one PyGOD dataset with one random seed (repository root)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("books", "disney", "enron", "reddit", "weibo")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--config", type=str, default=None, help="YAML path (default: configs/<dataset>.yaml)")
    p.add_argument("--num-trial", type=int, default=None, dest="num_trial")
    p.add_argument("--result-file", type=str, default=None)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    cfg = args.config or str(REPO / "configs" / f"{args.dataset}.yaml")
    cmd = [
        sys.executable,
        str(REPO / "main_train.py"),
        "--config",
        cfg,
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
    ]
    if args.num_trial is not None:
        cmd.extend(["--num_trial", str(args.num_trial)])
    if args.result_file:
        cmd.extend(["--result-file", args.result_file])
    if args.deterministic:
        cmd.append("--deterministic")
    env = os.environ.copy()
    if not env.get("FMGAD_MODEL_ROOT"):
        default_models = (REPO / "outputs" / "models").resolve()
        try:
            default_models.mkdir(parents=True, exist_ok=True)
            env["FMGAD_MODEL_ROOT"] = str(default_models)
        except OSError:
            pass
    return int(subprocess.call(cmd, cwd=str(REPO), env=env))


if __name__ == "__main__":
    raise SystemExit(main())
