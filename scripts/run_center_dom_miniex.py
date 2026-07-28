#!/usr/bin/env python3
"""Strict center-dom dual-flow control (books / disney).

Fixes vs previous miniex:
  - AE.eval() + freeze after load; latents cached once
  - center weights use cos(h, c) only (no residual fallback)
  - shuffled: fixed one-time permutation via independent Generator
  - polarity_enabled=False (report pre-polarity AUROC; smoothing kept)
  - per-seed config filenames (avoid concurrent YAML clobber)

Modes: uniform | learned | shuffled | reversed
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = ("books", "disney")
SEEDS = (0, 1, 2, 3, 42)
MODES = ("uniform", "learned", "shuffled", "reversed")
GUIDANCE_WEIGHT = 1.25
KAPPA_Q = 0.5
TAU_Q = 0.2


def _run_one(mode: str, dataset: str, seed: int, gpu: int, out_root: Path) -> Dict[str, Any]:
    result_path = out_root / "results" / mode / f"{dataset}_seed{seed}.json"
    log_path = out_root / "logs" / mode / f"{dataset}_seed{seed}.log"
    # Per-seed config path avoids concurrent writers clobbering the same YAML.
    cfg_path = out_root / "configs" / mode / f"{dataset}_seed{seed}.yaml"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {"mode": mode, "dataset": dataset, "seed": seed, "gpu": gpu}
    if result_path.exists():
        try:
            payload = json.load(open(result_path))
            stats = payload.get("dom_weight_stats") or {}
            row.update(
                {
                    "returncode": 0,
                    "auc": float(payload.get("auc_mean", payload.get("auc", float("nan")))),
                    "ap": float(payload.get("ap_mean", payload.get("ap", float("nan")))),
                    "n_eff": float(payload.get("n_eff", float("nan"))),
                    "n_eff_ratio": float(payload.get("n_eff_ratio", float("nan"))),
                    "center_valid": stats.get("center_valid"),
                    "feat_used": stats.get("feat_used"),
                    "s_std": stats.get("s_std"),
                    "elapsed_sec": 0.0,
                    "cached": True,
                }
            )
            print(
                f"[{mode}] {dataset} s{seed} cached auc={row['auc']:.4f} "
                f"N_eff_ratio={row['n_eff_ratio']:.3f} feat={row['feat_used']} "
                f"valid={row['center_valid']}",
                flush=True,
            )
            return row
        except Exception:
            pass

    cfg = yaml.safe_load(open(REPO / "configs" / f"{dataset}.yaml")) or {}
    cfg.update(
        {
            "dataset": dataset,
            "dual_flow_mode": "center_dom",
            "dom_weight_mode": mode,
            "dom_kappa_q": KAPPA_Q,
            "dom_tau_q": TAU_Q,
            "dom_control_seed": int(seed) + 100003,
            "allow_center_fallback": False,
            "use_proto": False,
            "use_proto_normal_weight": False,
            "weight": GUIDANCE_WEIGHT,
            "sample_steps": 1,
            "num_trial": 1,
            # Isolate dominant-weighting effect from polarity correction.
            "polarity_enabled": False,
            "exp_tag": f"center_dom_strict_{mode}_{dataset}",
        }
    )
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["FMGAD_MODEL_ROOT"] = str((out_root / "models").resolve())
    env["FMGAD_RUN_TAG_SUFFIX"] = f"{mode}_s{seed}"
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
        stats = payload.get("dom_weight_stats") or {}
        row["auc"] = float(payload.get("auc_mean", payload.get("auc", float("nan"))))
        row["ap"] = float(payload.get("ap_mean", payload.get("ap", float("nan"))))
        row["n_eff"] = float(payload.get("n_eff", float("nan")))
        row["n_eff_ratio"] = float(payload.get("n_eff_ratio", float("nan")))
        row["center_valid"] = stats.get("center_valid")
        row["feat_used"] = stats.get("feat_used")
        row["s_std"] = stats.get("s_std")
    else:
        row["auc"] = float("nan")
        row["ap"] = float("nan")
        row["n_eff"] = float("nan")
        row["n_eff_ratio"] = float("nan")
        row["center_valid"] = None
        row["feat_used"] = None
        row["s_std"] = None
    print(
        f"[{mode}] {dataset} s{seed} gpu={gpu} rc={rc} "
        f"auc={row['auc']:.4f} N_eff_ratio={row.get('n_eff_ratio', float('nan')):.3f} "
        f"feat={row.get('feat_used')} valid={row.get('center_valid')} "
        f"s_std={row.get('s_std')} {elapsed:.0f}s",
        flush=True,
    )
    return row


def _aggregate(out_root: Path) -> None:
    lines = [
        "# Strict center-dom controls (books / disney)",
        "",
        f"Guidance `w={GUIDANCE_WEIGHT}`, `κ_q={KAPPA_Q}`, `τ_q={TAU_Q}`. Seeds `{list(SEEDS)}`.",
        "",
        "Protocol: AE frozen eval; latents+π cached once; no residual fallback; "
        "`polarity_enabled=False`; shuffled = one fixed permutation.",
        "",
        "Metric: **pre-polarity AUROC** (score smoothing from yaml kept).",
        "",
        "| Mode | Books AUROC | Disney AUROC | Books N_eff/N | Disney N_eff/N | center_valid |",
        "|---|---:|---:|---:|---:|---|",
    ]
    summary: Dict[str, Any] = {}
    for mode in MODES:
        cells = []
        neff_cells = []
        valid_flags = []
        feats = []
        per = {}
        for ds in DATASETS:
            aucs, ratios = [], []
            for seed in SEEDS:
                p = out_root / "results" / mode / f"{ds}_seed{seed}.json"
                if not p.exists():
                    continue
                d = json.load(open(p))
                aucs.append(float(d.get("auc_mean", d.get("auc"))))
                ratios.append(float(d.get("n_eff_ratio", float("nan"))))
                st = d.get("dom_weight_stats") or {}
                valid_flags.append(st.get("center_valid"))
                feats.append(st.get("feat_used"))
            if len(aucs) == len(SEEDS):
                m, s = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
                cells.append(f"{m:.4f}±{s:.4f}")
                rr = [x for x in ratios if np.isfinite(x)]
                neff_cells.append(f"{float(np.mean(rr)):.3f}" if rr else "nan")
                per[ds] = {
                    "auc_mean": m,
                    "auc_std": s,
                    "n_eff_ratio_mean": float(np.mean(rr)) if rr else None,
                }
            else:
                cells.append(f"nan({len(aucs)}/5)")
                neff_cells.append("nan")
                per[ds] = {"n": len(aucs)}
        # center_valid summary: ignore uniform's trivial True/None
        if mode == "uniform":
            vstr = "n/a"
        else:
            bools = [x for x in valid_flags if x is not None]
            if not bools:
                vstr = "unknown"
            elif all(bools):
                vstr = "yes"
            elif not any(bools):
                vstr = "NO (collapsed)"
            else:
                vstr = f"mixed ({sum(bools)}/{len(bools)})"
        lines.append(
            f"| {mode} | {cells[0]} | {cells[1]} | {neff_cells[0]} | {neff_cells[1]} | {vstr} |"
        )
        summary[mode] = {"per_dataset": per, "center_valid": vstr, "feat_used_samples": feats[:4]}

    lines += ["", "## Hypothesis check (learned > others)", ""]
    for ds in DATASETS:
        learned = summary.get("learned", {}).get("per_dataset", {}).get(ds, {}).get("auc_mean")
        if learned is None:
            lines.append(f"- **{ds}**: incomplete")
            continue
        bits = []
        for other in ("uniform", "shuffled", "reversed"):
            o = summary.get(other, {}).get("per_dataset", {}).get(ds, {}).get("auc_mean")
            if o is None:
                bits.append(f"vs {other}=incomplete")
            else:
                ok = learned > o
                bits.append(f"vs {other}: {'PASS' if ok else 'FAIL'} (Δ={learned - o:+.4f})")
        lines.append(f"- **{ds}**: " + "; ".join(bits))

    lines += [
        "",
        "## Notes",
        "",
        "- If `center_valid=NO`, cosine(h,c) collapsed → weights fell back to uniform "
        "and the center-consistency hypothesis is **not tested** on that run.",
        "- `N_eff/N≈0.76` mainly reflects κ,τ rank-sigmoid sharpness, not center quality.",
        "",
    ]

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(summary_dir / "summary.json", "w"), indent=2)
    print("\n".join(lines), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", type=str, default="0,2,7")
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--output-dir", type=str, default="results/center_dom_strict")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--modes", type=str, default=",".join(MODES))
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    out_root = (REPO / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    if args.aggregate_only:
        _aggregate(out_root)
        return 0

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs: List[Tuple[str, str, int]] = []
    for mode in modes:
        for ds in datasets:
            for seed in seeds:
                jobs.append((mode, ds, seed))
    print(f"Jobs={len(jobs)} on GPUs {gpus} (modes={modes}, datasets={datasets})", flush=True)

    q: Queue = Queue()
    for i in range(args.max_workers):
        q.put(gpus[i % len(gpus)])

    def _wrap(job):
        mode, ds, seed = job
        gpu = q.get()
        try:
            return _run_one(mode, ds, seed, gpu, out_root)
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
