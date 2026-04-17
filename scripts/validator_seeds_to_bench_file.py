#!/usr/bin/env python3
"""
Convert a validator benchmark JSON (flat ``seeds`` list) into a bench ``--seed-file``
shape: ``{ "type1_city": [...], ... }`` using ``random_task`` challenge types.

Example:
  python3 scripts/validator_seeds_to_bench_file.py \\
    tuning_runs/complete-1.json \\
    --out tuning_runs/complete-1_bench_seeds.json

Then:
  python3 -m swarm.cli benchmark --model submission.zip \\
    --seed-file tuning_runs/complete-1_bench_seeds.json --workers 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _extract_seed_list(raw: Dict[str, Any]) -> List[int]:
    if "seeds" in raw and isinstance(raw["seeds"], list):
        return [int(x) for x in raw["seeds"]]
    raise ValueError("Expected top-level 'seeds' array (validator-style JSON).")


def main() -> None:
    p = argparse.ArgumentParser(description="Validator flat seeds -> swarm benchmark seed-file JSON.")
    p.add_argument("validator_json", type=Path, help="E.g. tuning_runs/complete-1.json")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for grouped seed file (use with benchmark --seed-file).",
    )
    args = p.parse_args()

    from swarm.benchmark.engine_parts.seeds import BENCH_GROUP_ORDER, _infer_bench_group
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task

    raw = json.loads(args.validator_json.read_text())
    flat = _extract_seed_list(raw)

    buckets: Dict[str, List[int]] = {g: [] for g in BENCH_GROUP_ORDER}
    unknown: List[int] = []

    for seed in flat:
        try:
            task = random_task(sim_dt=SIM_DT, seed=seed)
            ct = int(task.challenge_type)
        except Exception:
            unknown.append(int(seed))
            continue
        group = _infer_bench_group(ct, int(seed))
        if group is None:
            unknown.append(int(seed))
            continue
        buckets[group].append(int(seed))

    for g in BENCH_GROUP_ORDER:
        if not buckets[g]:
            raise SystemExit(f"No seeds mapped to {g}; cannot write bench file.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ordered = {g: buckets[g] for g in BENCH_GROUP_ORDER}
    args.out.write_text(json.dumps(ordered, indent=2) + "\n")

    counts = {g: len(buckets[g]) for g in BENCH_GROUP_ORDER}
    print(f"Wrote {args.out} ({sum(counts.values())} seeds). Per-group: {counts}")
    if unknown:
        print(f"WARNING: {len(unknown)} seeds skipped (no challenge type)", file=sys.stderr)


if __name__ == "__main__":
    main()
