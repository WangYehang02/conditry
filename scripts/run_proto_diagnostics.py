#!/usr/bin/env python3
"""Low-cost prototype diagnostics (guidance gap / controls / stability / multi-proto).

Forces weight=1.25 so proto guidance is active on all five datasets.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
SEEDS = (0, 1, 2, 3, 42)
GUIDANCE_WEIGHT = 1.25

# Phase order:
#  1) learned train+eval (also records D_guide + prototype)
#  2) inference-only controls reusing checkpoints
#  3) multi-proto M=2/4 (retrains proto branch)
CONTROL_MODES = ("shuffle", "random", "zero", "none")
MULTI_MS = (2, 4)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _result_path(out_root: Path, tag: str, dataset: str, seed: int) -> Path:
    return out_root / "results" / tag / f"{dataset}_seed{seed}.json"


def _run_job(
    tag: str,
    dataset: str,
    seed: int,
    gpu: int,
    out_root: Path,
    overrides: Dict[str, Any],
    env_extra: Dict[str, str],
    reuse: bool,
    deterministic: bool,
) -> Dict[str, Any]:
    result_path = _result_path(out_root, tag, dataset, seed)
    log_path = out_root / "logs" / tag / f"{dataset}_seed{seed}.log"
    cfg_path = out_root / "configs" / tag / f"{dataset}.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("auc_mean") is not None or payload.get("auc") is not None:
                payload["tag"] = tag
                payload["returncode"] = 0
                payload["cached"] = True
                print(
                    f"[{tag}] {dataset} s{seed} cached auc={payload.get('auc_mean', payload.get('auc'))}",
                    flush=True,
                )
                return payload
        except Exception:
            pass

    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    cfg.update(overrides)
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    cfg["exp_tag"] = f"pdiag_{dataset}"
    _save_yaml(cfg_path, cfg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    # Shared run tag so control modes reuse the learned checkpoints.
    env["FMGAD_RUN_TAG_SUFFIX"] = f"base_s{seed}"
    if reuse:
        env["FMGAD_REUSE_CHECKPOINTS"] = "1"
    else:
        env.pop("FMGAD_REUSE_CHECKPOINTS", None)
    env.update(env_extra)

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
    ]
    if deterministic:
        cmd.append("--deterministic")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT)
    payload: Dict[str, Any] = {
        "tag": tag,
        "dataset": dataset,
        "seed": seed,
        "returncode": rc,
        "elapsed_sec": time.time() - t0,
        "cached": False,
    }
    if rc == 0 and result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            payload.update(json.load(f))
    print(
        f"[{tag}] {dataset} s{seed} gpu={gpu} rc={rc} "
        f"auc={payload.get('auc_mean')} gap={payload.get('guidance_gap')} "
        f"{payload['elapsed_sec']:.0f}s",
        flush=True,
    )
    return payload


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _proto_stability(protos: List[np.ndarray]) -> float:
    """Mean pairwise cosine over seeds."""
    vecs = [p.reshape(-1) for p in protos if p is not None and p.size > 0]
    r = len(vecs)
    if r < 2:
        return float("nan")
    s = 0.0
    n = 0
    for i in range(r):
        for j in range(i + 1, r):
            a = vecs[i]
            b = vecs[j]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                cos = 0.0
            else:
                cos = float(np.dot(a, b) / (na * nb))
            s += cos
            n += 1
    return s / max(n, 1)


def _summarize(out_root: Path, rows: List[Dict[str, Any]]) -> Path:
    by: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for r in rows:
        if r.get("returncode", 1) != 0:
            continue
        by.setdefault(r["tag"], {}).setdefault(r["dataset"], []).append(r)

    lines = [
        "# Prototype diagnostics (weight forced to 1.25)",
        "",
        "## 1) Guidance gap \(D_{guide}=\mathbb{E}\|v^{free}-v^{proto}\|_2\)",
        "",
        "| Dataset | mean±std |",
        "|---|---:|",
    ]
    gap_table = {}
    for ds in DATASETS:
        gaps = [float(r["guidance_gap"]) for r in by.get("learned", {}).get(ds, []) if r.get("guidance_gap") is not None]
        m, s = _mean_std(gaps)
        gap_table[ds] = {"mean": m, "std": s, "n": len(gaps)}
        lines.append(f"| {ds} | {m:.4f}±{s:.4f} (n={len(gaps)}) |")

    lines.extend(
        [
            "",
            "## 2) Prototype control AUROC (reuse checkpoints; inference-only overrides)",
            "",
            "| Mode | Books | Disney | Enron | Reddit | Weibo | Avg |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    control_tags = ["learned", "shuffle", "random", "zero", "none"] + [f"m{m}" for m in MULTI_MS]
    auc_summary: Dict[str, Dict[str, Any]] = {}
    for tag in control_tags:
        cells = []
        means = []
        auc_summary[tag] = {}
        for ds in DATASETS:
            aucs = [
                float(r.get("auc_mean", r.get("auc")))
                for r in by.get(tag, {}).get(ds, [])
                if r.get("auc_mean") is not None or r.get("auc") is not None
            ]
            m, s = _mean_std(aucs)
            auc_summary[tag][ds] = {"mean": m, "std": s, "n": len(aucs)}
            if len(aucs) == 0:
                cells.append("nan")
            else:
                cells.append(f"{m:.3f}")
                means.append(m)
        avg = float(np.mean(means)) if means else float("nan")
        auc_summary[tag]["avg"] = avg
        lines.append(f"| `{tag}` | " + " | ".join(cells) + f" | **{avg:.3f}** |")

    lines.extend(
        [
            "",
            "## 3) Prototype stability \(S_{proto}\) (pairwise cosine over seeds)",
            "",
            "| Dataset | S_proto |",
            "|---|---:|",
        ]
    )
    stab = {}
    for ds in DATASETS:
        protos = []
        for r in by.get("learned", {}).get(ds, []):
            p = r.get("prototype")
            if p is not None:
                protos.append(np.asarray(p, dtype=np.float64))
        s = _proto_stability(protos)
        stab[ds] = s
        lines.append(f"| {ds} | {s:.4f} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All runs force `weight=1.25` (Enron/Reddit/Weibo default yaml has weight=0).",
            "- `shuffle/random/zero/none` reuse AE+FM checkpoints from `learned`.",
            "- `m2/m4` retrain the proto branch with soft k-means contexts on residual/z features.",
            "- If \(D_{guide}\\approx 0\), changing \(w\) cannot help on that dataset.",
        ]
    )

    summary = {
        "guidance_weight": GUIDANCE_WEIGHT,
        "guidance_gap": gap_table,
        "auc": auc_summary,
        "proto_stability": stab,
        "n_rows": len(rows),
    }
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    md_path = summary_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(lines), flush=True)
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=str, default="2,4,7")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--output-dir", type=str, default="results/proto_diagnostics")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--skip-multi", action="store_true")
    ap.add_argument("--deterministic", action="store_true", default=True)
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    base_overrides = {
        "weight": GUIDANCE_WEIGHT,
        "use_proto": True,
        "use_proto_normal_weight": False,  # clean uniform proto CFM for diagnostics
        "num_trial": 1,
    }

    jobs: List[Tuple] = []
    idx = 0

    def _add(tag, ds, seed, overrides, env_extra, reuse):
        nonlocal idx
        jobs.append((tag, ds, seed, gpus[idx % len(gpus)], overrides, env_extra, reuse))
        idx += 1

    # Phase 1: learned (train)
    for ds in datasets:
        for seed in seeds:
            _add(
                "learned",
                ds,
                seed,
                deepcopy(base_overrides),
                {"FMGAD_PROTO_MODE": "learned", "FMGAD_PROTO_M": "1"},
                False,
            )

    # Phase 2: controls (reuse)
    for mode in CONTROL_MODES:
        for ds in datasets:
            for seed in seeds:
                _add(
                    mode,
                    ds,
                    seed,
                    deepcopy(base_overrides),
                    {"FMGAD_PROTO_MODE": mode, "FMGAD_PROTO_M": "1"},
                    True,
                )

    # Phase 3: multi-proto (retrain proto; can reuse AE/free if present)
    if not args.skip_multi:
        for m in MULTI_MS:
            for ds in datasets:
                for seed in seeds:
                    _add(
                        f"m{m}",
                        ds,
                        seed,
                        deepcopy(base_overrides),
                        {"FMGAD_PROTO_MODE": "learned", "FMGAD_PROTO_M": str(m)},
                        True,  # reuse AE/free; proto ckpt will be overwritten by train if missing M-specific path
                    )

    print(f"Scheduling {len(jobs)} diagnostic jobs on GPUs {gpus}", flush=True)

    # Run learned first (no reuse), then the rest.
    def _execute(batch: List[Tuple], max_workers: int) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch), len(gpus))) as ex:
            futs = [
                ex.submit(
                    _run_job,
                    tag,
                    ds,
                    seed,
                    gpu,
                    out_root,
                    overrides,
                    env_extra,
                    reuse,
                    bool(args.deterministic),
                )
                for (tag, ds, seed, gpu, overrides, env_extra, reuse) in batch
            ]
            for fut in as_completed(futs):
                out_rows.append(fut.result())
        return out_rows

    learned_jobs = [j for j in jobs if j[0] == "learned"]
    other_jobs = [j for j in jobs if j[0] != "learned"]
    rows: List[Dict[str, Any]] = []
    rows.extend(_execute(learned_jobs, args.max_workers))
    rows.extend(_execute(other_jobs, args.max_workers))

    with open(out_root / "all_rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _summarize(out_root, rows)
    fails = sum(1 for r in rows if r.get("returncode", 1) != 0)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
