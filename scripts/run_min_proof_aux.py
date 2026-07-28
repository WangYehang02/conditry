#!/usr/bin/env python3
"""Minimum proof ablation (books / disney only).

Compare:
  1. single_flow          — no auxiliary branch
  2. duplicate_aux        — second flow, same objective, independent init
  3. current_proto        — current proto-conditioned auxiliary branch
  4. wider_single         — param-matched wider single flow

Goal: show current_proto beats (1)(2)(4) if the proto branch is doing real work.
Uses yaml polarity / smoothing (preserve reported Books/Disney performance setup).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = ("books", "disney")
SEEDS = (0, 1, 2, 3, 42)
GUIDANCE_WEIGHT = 1.25

VARIANTS: Dict[str, Dict[str, Any]] = {
    "single_flow": {
        "label": "Single flow",
        "role": "没有辅助分支",
        "overrides": {
            "dual_flow_mode": "single_full",
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": 0.0,
        },
    },
    "duplicate_aux": {
        "label": "Duplicate auxiliary flow",
        "role": "第二个 flow 与第一个目标相同，只是独立初始化",
        "overrides": {
            "dual_flow_mode": "duplicate_dual",
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": GUIDANCE_WEIGHT,
        },
    },
    "current_proto": {
        "label": "Current auxiliary flow",
        "role": "当前 proto-conditioned 分支",
        "overrides": {
            "dual_flow_mode": None,
            "use_proto": True,
            "use_proto_normal_weight": False,
            # keep yaml weight (disney=1.25, books=1.25)
        },
    },
    "wider_single": {
        "label": "Wider single flow",
        "role": "参数量匹配",
        "overrides": {
            "dual_flow_mode": "wider_single",
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": 0.0,
        },
    },
}
VARIANT_ORDER = list(VARIANTS.keys())


def _run_one(variant: str, dataset: str, seed: int, gpu: int, out_root: Path) -> Dict[str, Any]:
    result_path = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / variant / f"{dataset}_seed{seed}.log"
    cfg_path = out_root / "configs" / variant / f"{dataset}_seed{seed}.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {"variant": variant, "dataset": dataset, "seed": seed, "gpu": gpu}
    if result_path.exists():
        try:
            payload = json.load(open(result_path))
            row.update(
                {
                    "returncode": 0,
                    "auc": float(payload.get("auc_mean", payload.get("auc", float("nan")))),
                    "ap": float(payload.get("ap_mean", payload.get("ap", float("nan")))),
                    "elapsed_sec": 0.0,
                    "cached": True,
                }
            )
            print(
                f"[{variant}] {dataset} s{seed} cached auc={row['auc']:.4f}",
                flush=True,
            )
            return row
        except Exception:
            pass

    cfg = yaml.safe_load(open(REPO / "configs" / f"{dataset}.yaml")) or {}
    cfg.update(deepcopy(VARIANTS[variant]["overrides"]))
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    cfg["num_trial"] = 1
    cfg["exp_tag"] = f"minproof_{variant}_{dataset}"
    # Keep polarity from yaml (True) to preserve reported performance protocol.
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{variant}_s{seed}"
    env.pop("FMGAD_REUSE_CHECKPOINTS", None)

    cmd = [
        PYTHON,
        str(REPO / "main_train.py"),
        "--config",
        str(cfg_path),
        "--seed",
        str(seed),
        "--device",
        "0",
        "--result-file",
        str(result_path),
        "--deterministic",
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    row["returncode"] = rc
    row["elapsed_sec"] = elapsed
    row["cached"] = False
    if rc == 0 and result_path.exists():
        payload = json.load(open(result_path))
        row["auc"] = float(payload.get("auc_mean", payload.get("auc", float("nan"))))
        row["ap"] = float(payload.get("ap_mean", payload.get("ap", float("nan"))))
    else:
        row["auc"] = float("nan")
        row["ap"] = float("nan")
    print(
        f"[{variant}] {dataset} s{seed} gpu={gpu} rc={rc} "
        f"auc={row['auc']:.4f} {elapsed:.0f}s",
        flush=True,
    )
    return row


def _aggregate(out_root: Path) -> None:
    lines = [
        "# Minimum proof (books / disney)",
        "",
        "Compare single / duplicate-aux / **current proto** / wider-single.",
        f"Guidance for dual variants: `w={GUIDANCE_WEIGHT}`. Seeds `{list(SEEDS)}`.",
        "Polarity / smoothing: keep dataset yaml (performance protocol).",
        "",
        "| Version | Role | Books AUROC | Disney AUROC | Avg |",
        "|---|---|---:|---:|---:|",
    ]
    summary: Dict[str, Any] = {}
    means_by_variant: Dict[str, float] = {}
    for variant in VARIANT_ORDER:
        cells = []
        means = []
        per = {}
        for ds in DATASETS:
            aucs = []
            for seed in SEEDS:
                p = out_root / "results" / variant / f"{ds}_seed{seed}.json"
                if p.exists():
                    d = json.load(open(p))
                    aucs.append(float(d.get("auc_mean", d.get("auc"))))
            if len(aucs) == len(SEEDS):
                m = float(np.mean(aucs))
                s = float(np.std(aucs, ddof=1))
                cells.append(f"{m:.4f}±{s:.4f}")
                means.append(m)
                per[ds] = {"auc_mean": m, "auc_std": s, "aucs": aucs}
            else:
                cells.append(f"nan({len(aucs)}/5)")
                per[ds] = {"n": len(aucs)}
        avg = float(np.mean(means)) if means else float("nan")
        means_by_variant[variant] = avg
        meta = VARIANTS[variant]
        bold = "**" if variant == "current_proto" else ""
        lines.append(
            f"| {bold}{meta['label']}{bold} | {meta['role']} | "
            f"{cells[0]} | {cells[1]} | {bold}{avg:.4f}{bold} |"
        )
        summary[variant] = {"label": meta["label"], "role": meta["role"], "per_dataset": per, "avg_auc": avg}

    cur = means_by_variant.get("current_proto")
    lines += ["", "## Minimum proof checks (current_proto > others)", ""]
    if cur is None or not np.isfinite(cur):
        lines.append("- incomplete")
    else:
        for other in ("single_flow", "duplicate_aux", "wider_single"):
            o = means_by_variant.get(other)
            if o is None or not np.isfinite(o):
                lines.append(f"- vs {other}: incomplete")
            else:
                ok = cur > o
                lines.append(
                    f"- vs `{other}`: {'PASS' if ok else 'FAIL'} "
                    f"(ΔAvg={cur - o:+.4f})"
                )
        for ds in DATASETS:
            cur_ds = summary["current_proto"]["per_dataset"].get(ds, {}).get("auc_mean")
            if cur_ds is None:
                continue
            bits = []
            for other in ("single_flow", "duplicate_aux", "wider_single"):
                o = summary[other]["per_dataset"].get(ds, {}).get("auc_mean")
                if o is None:
                    bits.append(f"{other}=na")
                else:
                    bits.append(f"{other}:{'PASS' if cur_ds > o else 'FAIL'}({cur_ds - o:+.4f})")
            lines.append(f"- **{ds}**: " + "; ".join(bits))

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(summary_dir / "summary.json", "w"), indent=2)
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=str, default="0,2,3,5,6,7")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--output-dir", type=str, default="results/min_proof_aux")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    if args.aggregate_only:
        _aggregate(out_root)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs: List[Tuple[str, str, int]] = []
    for v in variants:
        for ds in datasets:
            for seed in seeds:
                jobs.append((v, ds, seed))
    print(f"Jobs={len(jobs)} on GPUs {gpus}", flush=True)

    q: Queue = Queue()
    for i in range(args.max_workers):
        q.put(gpus[i % len(gpus)])

    def _wrap(job):
        v, ds, seed = job
        gpu = q.get()
        try:
            return _run_one(v, ds, seed, gpu, out_root)
        finally:
            q.put(gpu)

    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as ex:
        futs = [ex.submit(_wrap, j) for j in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            if row.get("returncode", 1) != 0:
                fails += 1

    _aggregate(out_root)
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'summary.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
