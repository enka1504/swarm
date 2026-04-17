#!/usr/bin/env python3
"""Local rollout evaluation for bench seeds (no Docker)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gym_pybullet_drones.utils.enums import ActionType

from scripts.generate_video import build_task
from swarm.constants import SIM_DT, SPEED_LIMIT


def _load_agent_from_dir(extracted: Path):
    prev_cwd = os.getcwd()
    os.chdir(str(extracted))
    try:
        spec = importlib.util.spec_from_file_location(
            "drone_agent_eval", str(extracted / "drone_agent.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load drone_agent.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agent = mod.DroneFlightController()
        for name in ("_ensure_loaded", "reset"):
            fn = getattr(agent, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        return agent
    finally:
        os.chdir(prev_cwd)


def _make_zip(drone_py: Path) -> Path:
    zpath = Path(tempfile.mkdtemp()) / "submission.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(drone_py, "drone_agent.py")
    return zpath


def _extract_and_load(drone_py: Path):
    zpath = _make_zip(drone_py)
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(work)
    return _load_agent_from_dir(work), work


def rollout_one(agent, seed: int, challenge_type: int) -> tuple[float, bool]:
    from swarm.utils.env_factory import make_env

    task = build_task(seed, challenge_type)
    env = make_env(task, gui=False)
    obs, _ = env.reset(seed=task.map_seed)
    agent.reset()
    t_sim = 0.0
    success = False
    score = 0.0
    act_lo = np.asarray(env.action_space.low, dtype=np.float32).flatten()
    act_hi = np.asarray(env.action_space.high, dtype=np.float32).flatten()

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
            score = float(info.get("score", 0.0))
            if terminated or truncated:
                success = bool(info.get("success", False))
                break
    finally:
        try:
            env.close()
        except Exception:
            pass

    return score, success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone-agent", type=Path, default=_REPO / "my_agent" / "drone_agent.py")
    ap.add_argument("--seed-file", type=Path, default=_REPO / "tuning_runs" / "bench_200_fixed.json")
    ap.add_argument(
        "--groups",
        default="type2_open,type3_mountain,type4_village",
        help="Comma-separated bench groups",
    )
    ap.add_argument("--max-seeds", type=int, default=0, help="Cap total seeds (0 = all)")
    ap.add_argument(
        "--per-group",
        type=int,
        default=0,
        help="If >0, take only this many seeds from each group (after --groups filter).",
    )
    args = ap.parse_args()

    raw = json.loads(args.seed_file.read_text())
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    type_map = {
        "type1_city": 1,
        "type2_open": 2,
        "type3_mountain": 3,
        "type4_village": 4,
        "type5_warehouse": 5,
        "type6_forest": 6,
    }
    jobs: list[tuple[str, int, int]] = []
    for g in groups:
        seeds = [int(s) for s in raw.get(g, [])]
        if args.per_group > 0:
            seeds = seeds[: args.per_group]
        for s in seeds:
            jobs.append((g, s, type_map[g]))

    if args.max_seeds > 0:
        jobs = jobs[: args.max_seeds]

    agent, work = _extract_and_load(args.drone_agent)
    scores = []
    succ = []
    try:
        for i, (g, seed, ct) in enumerate(jobs):
            sc, ok = rollout_one(agent, seed, ct)
            scores.append(sc)
            succ.append(ok)
            print(f"[{i+1}/{len(jobs)}] {g} seed={seed} score={sc:.4f} success={ok}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if not scores:
        print("No seeds")
        return
    mean = float(np.mean(scores))
    print(f"--- mean_score={mean:.4f} success_rate={float(np.mean(succ)):.4f} n={len(scores)}")


if __name__ == "__main__":
    main()
