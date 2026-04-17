#!/usr/bin/env python3
"""Evaluate agent on first N seeds from a validator batch file.

Reports screening checkpoints at 50/100/150/200 seeds.
Shows fail reason (COLLISION/TIMEOUT) and success time for each seed.

Usage:
    python3 scripts/eval_validator_batch.py \
        --drone-agent my_agent/drone_agent.py \
        --seed-file tuning_runs/epoch_1_batch_validator1.json \
        --num-seeds 100 --workers 6
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gym_pybullet_drones.utils.enums import ActionType
from scripts.generate_video import build_task
from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.validator.reward import flight_reward

# Screening checkpoint config
CHECKPOINTS = [50, 100, 150, 200]
EARLY_FAIL_FACTORS = {50: 0.60, 100: 0.80, 150: 0.90}

TYPE_NAMES = {1: "city", 2: "open", 3: "mountain", 4: "village", 5: "warehouse", 6: "forest"}


def seed_to_challenge_type(seed: int) -> int:
    """Reproduce validator logic: deterministic type from seed."""
    types = [1, 2, 3, 4, 5, 6]
    weights = [1/6] * 6
    rng = random.Random(seed + 999999)
    return rng.choices(types, weights=weights, k=1)[0]


def rollout_one_detailed(agent, seed: int, challenge_type: int, force_moving: bool = False) -> dict:
    """Run one seed and return detailed result dict."""
    from swarm.utils.env_factory import make_env

    task = build_task(seed, challenge_type)
    if force_moving:
        task.moving_platform = True
    env = make_env(task, gui=False)
    obs, _ = env.reset(seed=task.map_seed)
    agent.reset()
    t_sim = 0.0
    success = False
    collision = False
    min_clearance = None
    act_lo = np.asarray(env.action_space.low, dtype=np.float32).flatten()
    act_hi = np.asarray(env.action_space.high, dtype=np.float32).flatten()
    info = {}

    try:
        while t_sim < task.horizon:
            try:
                raw = agent.act(obs)
                if raw is None:
                    raw = np.zeros(5, dtype=np.float32)
            except Exception:
                raw = np.zeros(5, dtype=np.float32)
            act = np.clip(np.asarray(raw, dtype=np.float32).flatten(), act_lo, act_hi)
            if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
                norm = max(float(np.linalg.norm(act[:3])), 1e-6)
                act[:3] *= min(1.0, float(SPEED_LIMIT) / norm)
                act = np.clip(act, act_lo, act_hi)

            obs, _, terminated, truncated, info = env.step(act[None, :])
            t_sim += float(SIM_DT)
            if terminated or truncated:
                success = bool(info.get("success", False))
                collision = bool(info.get("collision", False))
                min_clearance = info.get("min_clearance", None)
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    score = flight_reward(
        success=success,
        t=t_sim,
        horizon=task.horizon,
        task=task,
        min_clearance=min_clearance,
        collision=collision,
        legitimate_model=True,
    )

    # Determine fail reason
    if success:
        reason = "OK"
    elif collision:
        reason = "COLLISION"
    else:
        reason = "TIMEOUT"

    return {
        "score": score,
        "success": success,
        "reason": reason,
        "time": round(t_sim, 2),
        "horizon": round(task.horizon, 2),
        "collision": collision,
        "min_clearance": round(min_clearance, 3) if min_clearance is not None else None,
    }


def _run_single_seed(args_tuple):
    """Run one seed in a worker process. Each process loads its own agent."""
    drone_agent_path, seed, challenge_type, idx, total, force_moving = args_tuple

    agent_dir = Path(drone_agent_path).resolve().parent
    work = Path(tempfile.mkdtemp())
    try:
        for f in agent_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, work / f.name)
        prev_cwd = os.getcwd()
        os.chdir(str(work))
        try:
            spec = importlib.util.spec_from_file_location(
                "drone_agent_eval", str(work / "drone_agent.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            agent = mod.DroneFlightController()
        finally:
            os.chdir(prev_cwd)

        result = rollout_one_detailed(agent, seed, challenge_type, force_moving=force_moving)
        return idx, seed, challenge_type, result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def print_checkpoint(n, scores, successes, type_scores, type_details, threshold):
    avg = float(np.mean(scores))
    sr = float(np.mean(successes))
    print()
    print(f"{'='*60}")
    print(f"  CHECKPOINT: {n} seeds")
    print(f"  Avg score:    {avg:.4f}")
    print(f"  Success rate: {sr:.4f} ({sum(successes)}/{n})")
    print(f"  Threshold:    {threshold:.4f}")

    if n in EARLY_FAIL_FACTORS:
        cutoff = threshold * EARLY_FAIL_FACTORS[n]
        status = "PASS" if avg >= cutoff else "EARLY FAIL"
        print(f"  Early cutoff: {cutoff:.4f} (threshold x {EARLY_FAIL_FACTORS[n]})")
        print(f"  Status:       {status}")
    elif n >= 200:
        status = "PASS" if avg >= threshold else "FAIL"
        print(f"  Status:       {status}")

    print(f"  Per-type:")
    for t in range(1, 7):
        ts = type_scores[t]
        details = type_details[t]
        if ts:
            n_ok = sum(1 for d in details if d["success"])
            n_collision = sum(1 for d in details if d["reason"] == "COLLISION")
            n_timeout = sum(1 for d in details if d["reason"] == "TIMEOUT")
            ok_times = [d["time"] for d in details if d["success"]]
            avg_time_str = f"avg_t={np.mean(ok_times):.1f}s" if ok_times else ""
            print(f"    {TYPE_NAMES[t]:12s}: avg={np.mean(ts):.4f} "
                  f"ok={n_ok} col={n_collision} tout={n_timeout} "
                  f"n={len(ts)} {avg_time_str}")
    print(f"{'='*60}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone-agent", type=Path, default=_REPO / "my_agent" / "drone_agent.py")
    ap.add_argument("--seed-file", type=Path, required=True)
    ap.add_argument("--num-seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.4951,
                    help="Screening threshold for pass/fail reporting")
    ap.add_argument("--force-moving", action="store_true",
                    help="Force moving platform on every seed")
    args = ap.parse_args()

    data = json.loads(args.seed_file.read_text())
    seeds = data["seeds"][:args.num_seeds]

    print(f"Agent: {args.drone_agent}")
    print(f"Seeds: {len(seeds)} from {args.seed_file.name}")
    print(f"Workers: {args.workers}")
    print(f"Threshold: {args.threshold}")

    # Show type distribution
    type_counts = {}
    for s in seeds:
        t = seed_to_challenge_type(s)
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Type distribution: {', '.join(f'{TYPE_NAMES[k]}={v}' for k, v in sorted(type_counts.items()))}")
    print("=" * 80)

    # Build jobs in original seed order
    jobs = []
    for i, seed in enumerate(seeds):
        ct = seed_to_challenge_type(seed)
        jobs.append((str(args.drone_agent), seed, ct, i, len(seeds), args.force_moving))

    # Results array indexed by original position
    results = [None] * len(seeds)
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_single_seed, job): job[3] for job in jobs}

        for future in as_completed(futures):
            idx, seed, ct, result = future.result()
            results[idx] = (seed, ct, result)
            completed += 1
            r = result
            if r["success"]:
                detail = f"t={r['time']:.1f}/{r['horizon']:.0f}s"
                if r["min_clearance"] is not None:
                    detail += f" clr={r['min_clearance']:.2f}m"
            else:
                detail = r["reason"]
                if r["reason"] == "COLLISION":
                    detail += f" t={r['time']:.1f}s"
                else:
                    detail += f" t={r['time']:.1f}/{r['horizon']:.0f}s"
            print(f"[{completed:3d}/{len(seeds)}] {TYPE_NAMES[ct]:10s} seed={seed} "
                  f"score={r['score']:.4f} {detail}", flush=True)

    # Process results in original order for checkpoint reporting
    scores = []
    successes = []
    type_scores = {t: [] for t in range(1, 7)}
    type_details = {t: [] for t in range(1, 7)}

    for i, (seed, ct, r) in enumerate(results):
        scores.append(r["score"])
        successes.append(r["success"])
        type_scores[ct].append(r["score"])
        type_details[ct].append(r)

        n = i + 1
        if n in CHECKPOINTS or n == len(seeds):
            print_checkpoint(n, scores, successes, type_scores, type_details, args.threshold)

    # Final summary
    n_col = sum(1 for _, _, r in results if r["reason"] == "COLLISION")
    n_tout = sum(1 for _, _, r in results if r["reason"] == "TIMEOUT")
    n_ok = sum(1 for _, _, r in results if r["success"])
    print(f"\nFINAL: avg={np.mean(scores):.4f} ok={n_ok} collision={n_col} timeout={n_tout} total={len(scores)}")


if __name__ == "__main__":
    main()
