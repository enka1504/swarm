#!/usr/bin/env python3
"""
Run the 200-seed fixed bench with *forced moving platform* on every seed (12 workers by default),
write summary JSON, and exit non-zero if any SN124-style running-average checkpoint fails.

Checkpoint means (strict): mean(score of first N seeds in bench order) > threshold
  N=50  -> > 0.1554
  N=100 -> > 0.2072
  N=150 -> > 0.2331
  N=200 -> > 0.2509

Bench order matches JSON group order (type1_city … type6_forest) and seed order within each group.

Usage:
  cd /path/to/swarm && zip -j /tmp/submission.zip my_agent/drone_agent.py
  python3 scripts/champion_moving_bench.py /tmp/submission.zip

If any checkpoint fails: fix my_agent/drone_agent.py and re-run from the beginning (full 200).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Moving-platform 200-seed bench + SN124 checkpoints")
    ap.add_argument(
        "model",
        type=Path,
        nargs="?",
        default=Path("/tmp/submission_drone.zip"),
        help="Submission zip path",
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--seed-file",
        type=Path,
        default=repo / "tuning_runs" / "bench_200_fixed.json",
    )
    ap.add_argument(
        "--summary-json-out",
        type=Path,
        default=repo / "tuning_runs" / "last_champion_moving_summary.json",
    )
    args = ap.parse_args()

    bench = repo / "scripts" / "bench_full_eval.py"
    if not bench.is_file():
        print(f"Missing {bench}", file=sys.stderr)
        sys.exit(1)
    if not args.model.is_file():
        print(f"Model zip not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(bench),
        "--model",
        str(args.model.resolve()),
        "--workers",
        str(max(1, int(args.workers))),
        "--seed-file",
        str(args.seed_file.resolve()),
        "--force-moving-platform",
        "--rpc-verbosity",
        "low",
        "--summary-json-out",
        str(args.summary_json_out.resolve()),
    ]
    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(repo))
    if r.returncode != 0:
        sys.exit(r.returncode)

    data = json.loads(args.summary_json_out.read_text())
    sc = data.get("screening_checkpoints") or {}
    failed = [
        (k, v)
        for k, v in sorted(sc.items())
        if isinstance(v, dict) and v.get("pass") is False
    ]
    if failed:
        print("\n=== CHAMPION CHECKPOINTS FAILED ===", file=sys.stderr)
        for k, v in failed:
            print(f"  {k}: {json.dumps(v)}", file=sys.stderr)
        sys.exit(2)

    print("\n=== All SN124 screening checkpoints PASSED (moving platform, 200 seeds) ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
