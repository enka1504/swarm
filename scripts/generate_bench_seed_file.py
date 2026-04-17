#!/usr/bin/env python3
"""
Generate a benchmark --seed-file JSON with a chosen total seed count.

The Swarm benchmark has six map groups. This script splits *total* seeds across
groups as evenly as possible (e.g. 200 → 34+34+33+33+33+33), then discovers valid
map seeds the same way as swarm.benchmark.engine_parts.seeds._find_seeds.

Example:
  python3 scripts/generate_bench_seed_file.py --total 200 --rng 42 \\
    --out tuning_runs/bench_200_seeds.json

Then:
  swarm benchmark --model submission.zip --seed-file tuning_runs/bench_200_seeds.json --workers 12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _per_group_targets(total: int) -> dict[str, int]:
    if total < 6:
        raise ValueError("total must be at least 6 (one seed per group).")
    base, rem = divmod(total, 6)
    order = [
        "type1_city",
        "type2_open",
        "type3_mountain",
        "type4_village",
        "type5_warehouse",
        "type6_forest",
    ]
    targets: dict[str, int] = {}
    for i, g in enumerate(order):
        targets[g] = base + (1 if i < rem else 0)
    return targets


def _find_seeds_per_group(targets: dict[str, int], rng_seed: int) -> dict[str, list[int]]:
    from swarm.benchmark.engine_parts.seeds import _infer_bench_group
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task

    random.seed(rng_seed)
    groups: dict[str, list[int]] = {g: [] for g in targets}

    start = random.randint(100000, 900000)
    seed = start
    max_search = start + 500000

    while seed < max_search:
        task = random_task(sim_dt=SIM_DT, seed=seed)
        group = _infer_bench_group(int(task.challenge_type), seed)
        if group is not None and group in groups and len(groups[group]) < targets[group]:
            groups[group].append(int(seed))

        if all(len(groups[g]) >= targets[g] for g in targets):
            break
        seed += 1

    missing = [g for g in targets if len(groups[g]) < targets[g]]
    if missing:
        raise RuntimeError(
            "Could not find enough seeds: "
            + ", ".join(f"{g} ({len(groups[g])}/{targets[g]})" for g in missing)
        )
    return groups


def main() -> None:
    p = argparse.ArgumentParser(description="Generate benchmark seed JSON (exact total across 6 groups).")
    p.add_argument("--total", type=int, default=200, help="Total seeds (split evenly across 6 groups).")
    p.add_argument("--rng", type=int, default=42, help="RNG seed for discovery (reproducible).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tuning_runs/bench_200_seeds.json"),
        help="Output JSON path.",
    )
    args = p.parse_args()

    targets = _per_group_targets(args.total)
    found = _find_seeds_per_group(targets, args.rng)
    # Stable key order for readability
    ordered = {g: found[g] for g in targets}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ordered, indent=2) + "\n")

    s = sum(len(v) for v in ordered.values())
    print(f"Wrote {args.out} ({s} seeds). Per-group targets: {dict(targets)}")
    print(f"Actual counts: {{{', '.join(f'{k}: {len(ordered[k])}' for k in ordered)}}}")


if __name__ == "__main__":
    main()
