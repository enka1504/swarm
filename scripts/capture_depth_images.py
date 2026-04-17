#!/usr/bin/env python3
"""Capture real depth images from each of the 5 main challenge types (excluding Open Flight)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image
from pathlib import Path

# Import environment factory and task builder
from scripts.generate_video import build_task
from swarm.utils.env_factory import make_env

CHALLENGE_TYPES = {
    1: "city",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}

SEED = 400042
OUT_DIR = Path("/root/swarm/depth_images")
OUT_DIR.mkdir(exist_ok=True)

# Number of steps to fly before capturing (so drone is in the environment, not just at start)
WARMUP_STEPS = 30


def capture_depth(challenge_type: int, label: str):
    print(f"Capturing type {challenge_type} ({label})...")
    task = build_task(seed=SEED, challenge_type=challenge_type)
    env = make_env(task, gui=False)
    obs, _ = env.reset(seed=task.map_seed)

    # Fly forward a bit to get an interesting view
    for _ in range(WARMUP_STEPS):
        # Action: [dir_x, dir_y, dir_z, speed, yaw] - fly forward
        action = np.array([[0.0, 1.0, 0.0, 0.5, 0.0]], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    depth = obs["depth"]  # (128, 128, 1), float32, [0,1]
    depth_2d = depth[:, :, 0]  # (128, 128)

    # Save as agent sees it: 0=close(dark), 1=far(light)
    img_agent = (depth_2d * 255).astype(np.uint8)
    img_agent_pil = Image.fromarray(img_agent, mode="L")
    img_agent_pil = img_agent_pil.resize((512, 512), Image.NEAREST)
    path_agent = OUT_DIR / f"type{challenge_type}_{label}_agent_view.png"
    img_agent_pil.save(path_agent)

    # Save as video shows it: inverted (close=light, far=dark)
    img_video = ((1.0 - depth_2d) * 255).astype(np.uint8)
    img_video_pil = Image.fromarray(img_video, mode="L")
    img_video_pil = img_video_pil.resize((512, 512), Image.NEAREST)
    path_video = OUT_DIR / f"type{challenge_type}_{label}_video_view.png"
    img_video_pil.save(path_video)

    # Stats
    print(f"  Mean brightness (agent): {depth_2d.mean():.3f}, "
          f"Min: {depth_2d.min():.3f}, Max: {depth_2d.max():.3f}")

    env.close()
    print(f"  Saved: {path_agent.name} and {path_video.name}")


if __name__ == "__main__":
    for ctype, label in CHALLENGE_TYPES.items():
        try:
            capture_depth(ctype, label)
        except Exception as e:
            print(f"  ERROR on type {ctype}: {e}")
    print(f"\nAll images saved to {OUT_DIR}")
