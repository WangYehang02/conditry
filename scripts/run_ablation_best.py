#!/usr/bin/env python3
"""Ablation study under current best per-dataset configs (5 seeds)."""
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
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
SEEDS = (0, 1, 2, 3, 42)

# Display names match the paper-style ablation table.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "full": {
        "label": "Full RALFlow-GAD",
        "overrides": {},
    },
    "wo_score_orientation": {
        "label": "w/o score orientation",
        "overrides": {"polarity_enabled": False},
    },
    "wo_smooth": {
        "label": "w/o Graph Score Smoothing",
        # Smoothing is always applied; alpha=0 makes it a no-op.
        "overrides": {"score_smoothing_alpha": 0.0},
    },
    "wo_residual": {
        "label": "w/o residual augmented",
        "overrides": {"residual_scale": 0.0},
    },
    "wo_residual_fusion": {
        # Degree-aware gate becomes alpha=sigmoid(0)=0.5 for all nodes.
        "label": "w/o Residual Fusion",
        "overrides": {"gate_sharpness": 0.0},
    },
    "wo_virtual": {
        "label": "w/o Virtual-Neighbor",
        "overrides": {"use_virtual_neighbors": False},
    },
    "wo_proto": {
        "label": "w/o Prototype Guidance",
        "overrides": {"use_proto": False},
    },
}

VARIANT_ORDER = list(VARIANTS.keys())


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _pick_auc(payload: Dict[str, Any]) -> float:
    for k in ("auc", "auc_mean", "AUC"):
        if k in payload and payload[k] is not None:
            return float(payload[k])
    raise KeyError(f"no auc in keys={list(payload.keys())}")


def _run_one(
    variant: str,
    dataset: str,
    seed: int,
    gpu: int,
    out_root: Path,
    deterministic: bool,
) -> Tuple[str, str, int, int, float]:
    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    cfg.update(deepcopy(VARIANTS[variant]["overrides"]))
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    cfg["exp_tag"] = f"ablation_{variant}_{dataset}"

    cfg_path = out_root / "configs" / variant / f"{dataset}.yaml"
    result_path = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / variant / f"{dataset}_seed{seed}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                auc = _pick_auc(json.load(f))
            return variant, dataset, seed, 0, auc
        except Exception:
            pass

    _save_yaml(cfg_path, cfg)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{variant}_s{seed}"
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
    auc = float("nan")
    if rc == 0 and result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            auc = _pick_auc(json.load(f))
    print(
        f"[{variant}] {dataset} s{seed} gpu={gpu} rc={rc} auc={auc:.4f} {time.time()-t0:.0f}s",
        flush=True,
    )
    return variant, dataset, seed, rc, auc


def _aggregate(out_root: Path) -> Path:
    rows: Dict[str, Dict[str, List[float]]] = {v: {d: [] for d in DATASETS} for v in VARIANT_ORDER}
    for variant in VARIANT_ORDER:
        for dataset in DATASETS:
            for seed in SEEDS:
                p = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
                if not p.exists():
                    continue
                with open(p, "r", encoding="utf-8") as f:
                    rows[variant][dataset].append(_pick_auc(json.load(f)))

    # Round each dataset mean to 3 decimals first; Avg/ΔAvg from those displayed cells
    # so the table is manually verifiable (avoids raw-mean vs displayed inconsistency).
    table_means: Dict[str, Dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        table_means[variant] = {}
        disp = []
        for dataset in DATASETS:
            vals = rows[variant][dataset]
            if len(vals) != len(SEEDS):
                table_means[variant][dataset] = float("nan")
            else:
                table_means[variant][dataset] = round(float(np.mean(vals)), 3)
                disp.append(table_means[variant][dataset])
        table_means[variant]["avg"] = round(sum(disp) / len(disp), 3) if disp else float("nan")

    full_avg = table_means["full"]["avg"]
    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ablation Study (best configs)",
        "",
        "Mean AUROC over seeds `{0,1,2,3,42}`. Each cell is rounded to 3 decimals; "
        "**Avg.** is the mean of the five displayed dataset values. "
        "Each variant removes one component from the full model.",
        "",
        "| Variant | Books | Disney | Enron | Reddit | Weibo | Avg. | $\\Delta$Avg. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    drops = {
        v: round(table_means[v]["avg"] - full_avg, 3)
        for v in VARIANT_ORDER[1:]
        if not np.isnan(table_means[v]["avg"]) and not np.isnan(full_avg)
    }
    min_drop = min(drops.values()) if drops else None
    for variant in VARIANT_ORDER:
        label = VARIANTS[variant]["label"]
        m = table_means[variant]
        if variant == "full":
            delta_s = "—"
            avg_s = f"**{m['avg']:.3f}**" if not np.isnan(m["avg"]) else "nan"
            name = f"**{label}**"
        else:
            delta = drops.get(variant, float("nan"))
            delta_s = f"{delta:.3f}" if not np.isnan(delta) else "nan"
            if min_drop is not None and not np.isnan(delta) and delta == min_drop:
                delta_s = f"**{delta:.3f}**"
            avg_s = f"{m['avg']:.3f}" if not np.isnan(m["avg"]) else "nan"
            name = label
        cells = [f"{m[d]:.3f}" if not np.isnan(m[d]) else "nan" for d in DATASETS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {avg_s} | {delta_s} |")

    lines.append("")
    lines.append("Variant definitions:")
    lines.append("- `w/o score orientation`: `polarity_enabled=false`")
    lines.append("- `w/o Graph Score Smoothing`: `score_smoothing_alpha=0` (smoothing permanently on; alpha=0 is a no-op)")
    lines.append("- `w/o residual augmented`: `residual_scale=0`")
    lines.append("- `w/o Residual Fusion`: `gate_sharpness=0` (equal local/global mix)")
    lines.append("- `w/o Virtual-Neighbor`: `use_virtual_neighbors=false`")
    lines.append("- `w/o Prototype Guidance`: `use_proto=false`")
    md_path = summary_dir / "ablation_table.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # LaTeX
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation study of RALFlow-GAD under the best per-dataset configurations. Values are mean AUROC over five seeds.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l|ccccc|cc}",
        r"\toprule",
        r"Variant & Books & Disney & Enron & Reddit & Weibo & Avg. & $\Delta$Avg. \\",
        r"\midrule",
    ]
    for variant in VARIANT_ORDER:
        label = VARIANTS[variant]["label"]
        m = table_means[variant]
        if variant == "full":
            row = (
                rf"\textbf{{{label}}} & "
                + " & ".join(f"{m[d]:.3f}" for d in DATASETS)
                + rf" & \textbf{{{m['avg']:.3f}}} & -- \\"
            )
        else:
            dlt = drops.get(variant, float("nan"))
            dlt_s = f"{dlt:.3f}" if not np.isnan(dlt) else "nan"
            row = (
                rf"{label} & "
                + " & ".join(f"{m[d]:.3f}" for d in DATASETS)
                + rf" & {m['avg']:.3f} & {dlt_s} \\"
            )
        tex.append(row)
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (summary_dir / "ablation_table.tex").write_text("\n".join(tex), encoding="utf-8")

    with open(summary_dir / "ablation_means.json", "w", encoding="utf-8") as f:
        json.dump(table_means, f, indent=2)
    print(md_path.read_text(encoding="utf-8"), flush=True)
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, default=str(REPO / "results" / "ablation_best"))
    ap.add_argument("--gpus", type=str, default="0,1,2,4")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(out_root)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    datasets = [x for x in args.datasets.split(",") if x.strip()]
    variants = [x for x in args.variants.split(",") if x.strip()]
    jobs = [(v, d, s) for v in variants for d in datasets for s in SEEDS]
    print(f"Scheduling {len(jobs)} jobs on GPUs {gpus}", flush=True)

    # Round-robin GPU assignment via a simple counter in threads is racy;
    # use a queue of gpu slots equal to max_workers.
    from queue import Queue

    gpu_q: Queue = Queue()
    for i in range(args.max_workers):
        gpu_q.put(gpus[i % len(gpus)])

    def _wrapped(job):
        v, d, s = job
        gpu = gpu_q.get()
        try:
            return _run_one(v, d, s, gpu, out_root, args.deterministic)
        finally:
            gpu_q.put(gpu)

    fails = 0
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as ex:
        futs = [ex.submit(_wrapped, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            _, _, _, rc, _ = fut.result()
            if rc != 0:
                fails += 1
            if i % 10 == 0 or i == len(futs):
                print(f"Progress {i}/{len(futs)} fails={fails}", flush=True)

    _aggregate(out_root)
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'ablation_table.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
