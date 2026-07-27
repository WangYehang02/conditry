#!/usr/bin/env python3
"""Fine-grained ablation of score orientation and prior fusion.

Variants (all other components/hyperparameters fixed to current best configs):
  flow_ranking_only   — raw Flow score (no polarity); AUROC == Flow ranking only
  orientation_only    — keep/flip by local_prior agreement; no prior fusion (η=1)
  prior_fusion_only   — blend unoriented Flow rank with local_prior (never flip)
  local_prior_only    — use local_prior rank only (η=0)
  full                — orientation + prior fusion (current best)

Reuses Full ablation checkpoints via FMGAD_REUSE_CHECKPOINTS (polarity is post-hoc).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
DATASETS = ("books", "disney", "enron", "reddit", "weibo")
SEEDS = (0, 1, 2, 3, 42)
FULL_CKPT_ROOT = REPO / "results" / "ablation_best"

VARIANTS: Dict[str, Dict[str, Any]] = {
    "flow_ranking_only": {
        "label": "Flow Ranking Only",
        "overrides": {"polarity_enabled": False},
        # Prefer already-finished identical runs from the component ablation.
        "copy_from": FULL_CKPT_ROOT / "results" / "wo_score_orientation",
    },
    "orientation_only": {
        "label": "Orientation Only",
        "overrides": {
            "polarity_enabled": True,
            "polarity_consensus_score_weight": 1.0,
        },
    },
    "prior_fusion_only": {
        "label": "Prior Fusion Only",
        # Never flip: agreement ∈ [-1,1], so threshold=-1.0 ⇒ flipped is always False.
        "overrides": {
            "polarity_enabled": True,
            "polarity_consensus_threshold": -1.0,
        },
    },
    "local_prior_only": {
        "label": "Local Prior Only",
        "overrides": {
            "polarity_enabled": True,
            "polarity_consensus_score_weight": 0.0,
        },
    },
    "full": {
        "label": "Full",
        "overrides": {},
        "copy_from": FULL_CKPT_ROOT / "results" / "full",
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


def _maybe_copy_existing(variant: str, dataset: str, seed: int, out_root: Path) -> bool:
    src_root = VARIANTS[variant].get("copy_from")
    if src_root is None:
        return False
    src = Path(src_root) / f"{dataset}_seed{seed}.json"
    dst = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return True


def _run_one(
    variant: str,
    dataset: str,
    seed: int,
    gpu: int,
    out_root: Path,
    deterministic: bool,
) -> Tuple[str, str, int, int, float]:
    result_path = out_root / "results" / variant / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / variant / f"{dataset}_seed{seed}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return variant, dataset, seed, 0, _pick_auc(json.load(f))
        except Exception:
            pass

    if _maybe_copy_existing(variant, dataset, seed, out_root):
        with open(result_path, "r", encoding="utf-8") as f:
            auc = _pick_auc(json.load(f))
        print(f"[{variant}] {dataset} s{seed} copied auc={auc:.4f}", flush=True)
        return variant, dataset, seed, 0, auc

    cfg = _load_yaml(REPO / "configs" / f"{dataset}.yaml")
    cfg.update(deepcopy(VARIANTS[variant]["overrides"]))
    cfg["dataset"] = dataset
    cfg["sample_steps"] = 1
    # Reuse Full checkpoint paths: ablation_full_{ds}_full_s{seed}
    cfg["exp_tag"] = f"ablation_full_{dataset}"

    cfg_path = out_root / "configs" / variant / f"{dataset}.yaml"
    _save_yaml(cfg_path, cfg)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((FULL_CKPT_ROOT / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"full_s{seed}"
    env["FMGAD_REUSE_CHECKPOINTS"] = "1"
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

    table: Dict[str, Dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        table[variant] = {}
        disp = []
        for dataset in DATASETS:
            vals = rows[variant][dataset]
            if len(vals) != len(SEEDS):
                table[variant][dataset] = float("nan")
            else:
                table[variant][dataset] = round(float(np.mean(vals)), 3)
                disp.append(table[variant][dataset])
        table[variant]["avg"] = round(sum(disp) / len(disp), 3) if disp else float("nan")

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Fine-grained score calibration ablation",
        "",
        "Mean AUROC over seeds `{0,1,2,3,42}`. Cells rounded to 3 decimals; "
        "**Avg.** is the mean of the five displayed dataset values.",
        "",
        "| Variant | Books | Disney | Enron | Reddit | Weibo | Avg. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        label = VARIANTS[variant]["label"]
        m = table[variant]
        cells = [f"{m[d]:.3f}" if not np.isnan(m[d]) else "--" for d in DATASETS]
        avg = f"{m['avg']:.3f}" if not np.isnan(m["avg"]) else "--"
        if variant == "full":
            lines.append(f"| **Full** | " + " | ".join(cells) + f" | **{avg}** |")
        else:
            lines.append(f"| {label} | " + " | ".join(cells) + f" | {avg} |")

    lines.extend(
        [
            "",
            "Variant definitions:",
            "- `Flow Ranking Only`: `polarity_enabled=false` (raw Flow score; AUROC-equivalent to ranking)",
            "- `Orientation Only`: keep/flip by local_prior agreement; `polarity_consensus_score_weight=1.0`",
            "- `Prior Fusion Only`: blend unoriented Flow rank with local_prior; `polarity_consensus_threshold=-1.0` (never flip)",
            "- `Local Prior Only`: `polarity_consensus_score_weight=0.0`",
            "- `Full`: orientation + prior fusion (best configs)",
            "",
        ]
    )
    md_path = summary_dir / "score_calibration_table.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        r"Fine-grained ablation of score orientation and prior fusion.",
        r"All other components and hyperparameters are kept fixed.",
        r"}",
        r"\label{tab:score_calibration_ablation}",
        r"\resizebox{0.8\textwidth}{!}{",
        r"\begin{tabular}{l|rrrrr|r}",
        r"\toprule",
        r"\textbf{Variant}",
        r"& \textbf{Books}",
        r"& \textbf{Disney}",
        r"& \textbf{Enron}",
        r"& \textbf{Reddit}",
        r"& \textbf{Weibo}",
        r"& \textbf{Avg.} \\",
        r"\midrule",
    ]
    for variant in VARIANT_ORDER:
        label = VARIANTS[variant]["label"]
        m = table[variant]
        cells = [f"{m[d]:.3f}" if not np.isnan(m[d]) else "--" for d in DATASETS]
        avg = f"{m['avg']:.3f}" if not np.isnan(m["avg"]) else "--"
        if variant == "full":
            tex.append(
                rf"Full \method & "
                + " & ".join(cells)
                + rf" & {avg} \\"
            )
        else:
            tex.append(rf"{label} & " + " & ".join(cells) + rf" & {avg} \\")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    (summary_dir / "score_calibration_table.tex").write_text("\n".join(tex), encoding="utf-8")
    with open(summary_dir / "score_calibration_means.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(md_path.read_text(encoding="utf-8"), flush=True)
    return md_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=str,
        default=str(REPO / "results" / "score_calibration_ablation"),
    )
    ap.add_argument("--gpus", type=str, default="0,3,4,7")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER))
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(out_root)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    jobs = [(v, d, s) for v in variants for d in datasets for s in SEEDS]
    print(f"Scheduling {len(jobs)} jobs on GPUs {gpus}", flush=True)

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
    print(f"Done. fails={fails}. Summary: {out_root / 'summary' / 'score_calibration_table.md'}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
