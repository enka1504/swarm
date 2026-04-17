#!/usr/bin/env python3
"""Run a moving-platform benchmark for open/mountain/village only."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from swarm.benchmark.engine_parts.config import _Tee, _active_runtime_overrides, _debug_profile_options, _ts
from swarm.benchmark.engine_parts.reporting import _print_results
from swarm.benchmark.engine_parts.seeds import _infer_uid_from_model_path
from swarm.benchmark.engine_parts.workers import _run_benchmark


GROUP_ORDER = ("type1_city", "type2_open", "type3_mountain", "type4_village", "type5_warehouse", "type6_forest")
CHECKPOINTS = (
    (50, 0.15),
    (100, 0.19),
    (150, 0.21),
    (200, 0.25),
)


def _load_seed_file(path: Path) -> dict[str, list[int]]:
    raw = json.loads(path.read_text())
    type_seeds: dict[str, list[int]] = {}
    non_empty_groups = 0
    for group in GROUP_ORDER:
        seeds = raw.get(group)
        if not isinstance(seeds, list):
            raise ValueError(f"Seed file group {group} must be a list")
        if seeds:
            non_empty_groups += 1
        type_seeds[group] = [int(seed) for seed in seeds]
    if non_empty_groups == 0:
        raise ValueError("Seed file must contain at least one non-empty group")
    return type_seeds


def _apply_rpc_verbosity(run_opts, level: str) -> None:
    if level == "low":
        run_opts.rpc_trace = False
        run_opts.rpc_trace_every = 1000
        run_opts.rpc_heartbeat_sec = 0.0
    elif level == "mid":
        run_opts.rpc_trace = True
        run_opts.rpc_trace_every = 250
        run_opts.rpc_heartbeat_sec = 120.0
    else:
        run_opts.rpc_trace = True
        run_opts.rpc_trace_every = 25
        run_opts.rpc_heartbeat_sec = 30.0


def _requested_checkpoints(scores: list[float]) -> dict[str, dict[str, float | int | bool | None]]:
    out: dict[str, dict[str, float | int | bool | None]] = {}
    for n_need, thresh in CHECKPOINTS:
        key = f"first_{n_need}_seeds"
        if len(scores) >= n_need:
            mean_score = float(sum(scores[:n_need]) / n_need)
            out[key] = {
                "n": n_need,
                "threshold": thresh,
                "mean_score": mean_score,
                "pass": mean_score > thresh,
            }
        else:
            out[key] = {
                "n": n_need,
                "threshold": thresh,
                "mean_score": None,
                "pass": None,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Moving-platform benchmark for open/mountain/village only")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=REPO / "tuning_runs" / "bench_omv_200_rng42.json",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=REPO / "tuning_runs" / "last_omv_moving_summary.json",
    )
    parser.add_argument("--log-out", type=Path, default=None)
    parser.add_argument(
        "--rpc-verbosity",
        choices=("low", "mid", "high"),
        default="low",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    type_seeds = _load_seed_file(args.seed_file.resolve())
    total_seeds = sum(len(v) for v in type_seeds.values())
    uid = _infer_uid_from_model_path(model_path)
    if uid is None:
        uid = 0

    run_opts = _debug_profile_options()
    _apply_rpc_verbosity(run_opts, args.rpc_verbosity)
    run_opts.force_moving_platform = True

    log_path = args.log_out.resolve() if args.log_out else None
    log_fh = None
    out_stream = sys.__stdout__
    err_stream = sys.__stderr__

    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = open(log_path, "w")
            sys.stdout = _Tee(sys.__stdout__, log_fh)
            sys.stderr = _Tee(sys.__stderr__, log_fh)

        print(f"[{_ts()}] === OMW MOVING BENCHMARK ===")
        print(f"[{_ts()}] Model: {model_path}")
        print(f"[{_ts()}] UID: {uid}")
        print(f"[{_ts()}] Workers requested: {max(1, int(args.workers))}")
        print(f"[{_ts()}] RPC verbosity: {args.rpc_verbosity}")
        runtime_overrides = _active_runtime_overrides()
        if runtime_overrides:
            print(f"[{_ts()}] Runtime overrides: {runtime_overrides}")
        print(f"[{_ts()}] Groups: {', '.join(GROUP_ORDER)}")
        print(f"[{_ts()}] Seed file: {args.seed_file.resolve()}")
        print(f"[{_ts()}] Total seeds: {total_seeds}")
        for group in GROUP_ORDER:
            print(f"  {group}: {type_seeds[group]}")
        print()
        print(f"[{_ts()}] Forcing moving goal platform on every seed")
        print()

        (
            task_meta,
            results,
            seed_times,
            seed_wall_by_key,
            seed_status_by_key,
            full_wall_by_key,
            batch_stats,
            elapsed,
            eval_start,
            launched_workers,
        ) = asyncio.run(
            _run_benchmark(
                model_path,
                uid,
                type_seeds,
                max(1, int(args.workers)),
                run_opts=run_opts,
            )
        )
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = out_stream
        sys.stderr = err_stream

    print(f"\n[{_ts()}] === RESULTS ===")
    summary = _print_results(
        task_meta,
        results,
        seed_times,
        seed_wall_by_key,
        seed_status_by_key,
        full_wall_by_key,
        batch_stats,
        elapsed,
        eval_start,
        launched_workers,
        host_parallelism="process",
    )

    requested = _requested_checkpoints(summary.get("scores_in_evaluation_order", []))
    summary["requested_checkpoints"] = requested

    print("  Requested checkpoints (mean of first N seeds in OMW order):")
    for n_need, thresh in CHECKPOINTS:
        key = f"first_{n_need}_seeds"
        row = requested[key]
        if row["mean_score"] is None:
            print(f"    {key}: skipped")
        else:
            pf = "PASS" if row["pass"] else "FAIL"
            print(f"    {key}: mean={row['mean_score']:.4f} (need >{thresh:.4f}) -> {pf}")

    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\n[{_ts()}] Wrote summary JSON: {args.summary_json_out.resolve()}")

    if log_fh is not None:
        try:
            log_fh.flush()
        finally:
            log_fh.close()


if __name__ == "__main__":
    main()
