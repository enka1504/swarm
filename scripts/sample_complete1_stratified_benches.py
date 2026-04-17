#!/usr/bin/env python3
"""
Draw disjoint stratified random samples from ``complete-1_bench_seeds.json`` (or any
grouped bench file) so every map group is represented in each split.

Splits default: 18, 50, 50, 50, 100, 200 (468 seeds total from the 1000).

Example:
  python3 scripts/sample_complete1_stratified_benches.py \\
    --source tuning_runs/complete-1_bench_seeds.json \\
    --out-dir tuning_runs/complete1_splits --rng 12345
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from swarm.benchmark.engine_parts.seeds import BENCH_GROUP_ORDER  # noqa: E402


def _take_stratified(
    pools: Dict[str, List[int]], ptr: Dict[str, int], n: int
) -> Dict[str, List[int]]:
    q, r = divmod(n, 6)
    out: Dict[str, List[int]] = {}
    for i, g in enumerate(BENCH_GROUP_ORDER):
        need = q + (1 if i < r else 0)
        start = ptr[g]
        end = start + need
        chunk = pools[g][start:end]
        if len(chunk) < need:
            raise RuntimeError(
                f"Not enough seeds left in {g}: need {need}, have {len(pools[g]) - start}"
            )
        out[g] = list(chunk)
        ptr[g] = end
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        type=Path,
        default=Path("tuning_runs/complete-1_bench_seeds.json"),
        help="Grouped bench JSON (six non-empty lists).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("tuning_runs/complete1_splits"))
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[18, 50, 50, 50, 100, 200],
        help="Split sizes (must sum to <= total seeds per group availability).",
    )
    p.add_argument("--rng", type=int, default=12345, help="Shuffle RNG seed (reproducible).")
    p.add_argument("--prefix", type=str, default="complete1_split", help="Output filename prefix.")
    args = p.parse_args()

    raw = json.loads(args.source.read_text())
    pools: Dict[str, List[int]] = {}
    for g in BENCH_GROUP_ORDER:
        if g not in raw or not isinstance(raw[g], list) or not raw[g]:
            raise SystemExit(f"Source missing or empty group: {g}")
        pools[g] = list(int(x) for x in raw[g])

    rng = random.Random(args.rng)
    for g in BENCH_GROUP_ORDER:
        rng.shuffle(pools[g])

    ptr = {g: 0 for g in BENCH_GROUP_ORDER}

    need_by_group = {g: 0 for g in BENCH_GROUP_ORDER}
    for sz in args.sizes:
        q, r = divmod(sz, 6)
        for i, gg in enumerate(BENCH_GROUP_ORDER):
            need_by_group[gg] += q + (1 if i < r else 0)
    for g in BENCH_GROUP_ORDER:
        if need_by_group[g] > len(pools[g]):
            raise SystemExit(
                f"Group {g}: need {need_by_group[g]} seeds across splits but only {len(pools[g])} in source."
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"rng": args.rng, "source": str(args.source), "splits": []}

    labels = ["18", "50a", "50b", "50c", "100", "200"]
    for idx, sz in enumerate(args.sizes):
        split = _take_stratified(pools, ptr, sz)
        label = labels[idx] if idx < len(labels) else str(sz)
        out_path = args.out_dir / f"{args.prefix}_{label}.json"
        out_path.write_text(json.dumps(split, indent=2) + "\n")
        counts = {g: len(split[g]) for g in BENCH_GROUP_ORDER}
        meta["splits"].append({"file": str(out_path), "n": sz, "per_group": counts})
        print(f"Wrote {out_path} ({sz} seeds) {counts}")

    meta_path = args.out_dir / f"{args.prefix}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
