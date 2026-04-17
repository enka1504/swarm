#!/usr/bin/env python3
"""One-shot refactor: RNG update, method renames, docstrings, complete reset()."""
from __future__ import annotations

import ast
import re
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "my_agent" / "drone_agent.py"


def _has_leading_docstring(node: ast.FunctionDef) -> bool:
    if not node.body:
        return False
    first = node.body[0]
    if not isinstance(first, ast.Expr):
        return False
    v = first.value
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return True
    return isinstance(v, ast.Str)  # pragma: no cover


def _doc_for_name(name: str, body_indent: int = 8) -> str:
    """Return a docstring block with opening/closing quotes at ``body_indent`` spaces.

    ``body_indent`` must match the indentation of the first statement inside the
    function (typically ``node.col_offset + 4`` from :class:`ast.FunctionDef`).
    Using a fixed 8 spaces breaks nested functions (e.g. helpers inside
    :meth:`_find_goal_platform_from_depth`), which need 12+ spaces.
    """
    pad = " " * body_indent
    docs: dict[str, str] = {
        "__init__": """Initialize episode-scoped state. The benchmark calls :meth:`reset` between seeds.

        Swarm instantiates one controller per submission; attributes hold the discrete
        state (0/1/3), waypoint queue, goal estimate, and search bookkeeping.
        """,
        "_read_position_xyz": """Return world position ``[x,y,z]`` (m) from ``observation['state'][0:3]``.""",
        "_read_velocity_xyz": """Return linear velocity ``[vx,vy,vz]`` (m/s) from ``state[6:9]`` when present.""",
        "_read_roll_pitch_rad": """Return roll and pitch (rad) from ``state[3:5]``.""",
        "_read_yaw_rad": """Return yaw (rad) from ``state[5]``, or ``None`` if missing.""",
        "_read_angular_velocity": """Return body-frame angular velocity ``[p,q,r]`` from ``state[9:12]``.""",
        "_read_search_area_offset": """Return the last three state scalars: offset vector to the search-area anchor.""",
        "_project_world_direction_to_pixel": """Map a **unit** world-space direction to depth image ``(row, col)``.

        Uses the camera chain ``R = Rz·Ry·Rx`` and pinhole projection into the FOV.
        Returns ``None`` if the ray does not project into the forward hemisphere.
        """,
        "_unproject_pixel_to_world_direction": """Return a **unit** world direction for depth pixel ``(r,c)`` (camera model).""",
        "_camera_forward_in_world": """Return the camera optical axis as a unit vector in world coordinates.""",
        "_depth_along_world_direction": """Sample normalized depth at the pixel corresponding to ``direction_world`` and convert to metres.

        Depth is mapped ``0.5 + 19.5 * raw`` for ``raw in [0,1]``.
        """,
        "_median_scene_depth_m": """Median of depth image in metres (full frame); used for coarse scene cues.""",
        "_is_forward_center_clear_depth": """True if the centre row strip in front of the drone has at least ``min_clear_m`` range.""",
        "_front_patch_obstacle_too_close": """True if the depth patch around the motion/look ray is closer than ``front_obstacle_min_depth_m``.""",
        "_min_depth_in_velocity_front_patch": """Minimum depth (m) in the front patch along velocity or forward body axis.""",
        "_lateral_close_obstacle_corridor_clear": """Heuristic: front corridor clear but left/right columns show close obstacles (narrow gap).""",
        "_movement_direction_pixel": """Pixel coordinates for the ray aligned with velocity (or forward if slow).""",
        "_detour_waypoint_from_obstacle_cc": """Plan a detour: find obstacle CC from front pixel, dilate forbidden mask, pick feasible 3D point toward ``final_target``.""",
        "_clearest_horizontal_toward_search_center": """Among valid rays, pick horizontal direction in the image plane closest to ``direction_to_center`` (top-k by angle).""",
        "_tilt_accel_scale": """Reduce commanded acceleration when |roll|/|pitch| approach ``MAX_SAFE_ROLL_PITCH_RAD``.""",
        "_angular_rate_accel_scale": """Reduce acceleration when roll/pitch rates are high (stability).""",
        "_approach_angle_speed_scale": """Slow down when current velocity is misaligned with desired direction (cornering).""",
        "_velocity_command_toward_point": """Build the 5-D action to steer toward ``target_point`` with deceleration near the goal.

        Outputs ``[vx,vy,vz, speed, yaw_norm]`` with yaw in ``[-1,1]`` as ``heading/pi``.
        """,
        "_world_search_center_from_offset": """Compute search-area centre: current position plus offset vector from state.""",
        "_waypoints_for_height_match": """Build 1–2 waypoints to reach search altitude (or elevated cruise) before exploration.""",
        "_sample_random_search_point": """Uniform random offset inside the search ellipsoid (clamped in z). Uses ``self.rng``.""",
        "_near_depth_bearing": """Average direction toward pixels with depth below ``max_depth_m`` (near-field blob).""",
        "_goal_candidate_distance_score": """Scalar score: horizontal distance to search centre plus weighted |dz| (open vs clutter weights).""",
        "_goal_in_search_bounds": """Whether a world position lies inside the allowed XY/Z window around ``search_area_center``.""",
        "_fuse_goal_observation": """Merge a new goal observation with the stored estimate (distance gating + score improvement).""",
        "_assign_detected_goal": """Overwrite stored goal position, direction, score, and reset motion EMAs.""",
        "_clear_goal_kinematics": """Clear goal motion smoothing state (EMA velocity and displacement).""",
        "_update_goal_kinematics_ema": """Update EMA of goal motion and horizontal velocity for moving-platform lead.""",
        "_lead_compensated_goal": """Return goal position with optional horizontal lead using estimated goal velocity.""",
        "_state3_open_approach_command": """State-3 controller tuned when ``scene_is_open_like`` (hover, staged descent).""",
        "_merge_priority_goal_candidate": """Priority-based goal association for multi-hypothesis tracking (lower priority wins).""",
        "_state3_straight_goal_approach_command": """Simpler state-3 two-stage hover-then-descend profile (horizontal vs vertical regime).""",
        "_fallback_goal_toward_search_center": """When edge segmentation fails in open scenes, search a window toward the search centre.""",
        "_find_goal_platform_from_depth": """Segment depth edges, grow similar-depth regions, score platforms vs search centre; optional fallback.""",
        "_batch_pixels_to_world_xyz": """Vectorised back-projection of pixel list to world points at sampled depths.""",
        "_bfs_edge_connected_component": """8-connected BFS on edge mask with depth similarity; returns size, centroid pixel, edge pixels.""",
        "_grow_similar_depth_region": """Flood-fill from edge pixels to include similar-depth interior for planarity checks.""",
        "_state1_motion_limits": """Keyword arguments for :meth:`_velocity_command_toward_point` during state 1 (search).""",
        "_state1_next_search_waypoint": """Next exploration target: random jitter in clutter/open or ring pattern when ``scene_is_open_like``.""",
        "_depth_suggests_open_scene": """Heuristic: high median depth, high centre patch depth, low spawn altitude ⇒ open-like arena.""",
        "act": """Compute one control step from the simulator observation dict.

        Updates goal detections, transitions 0→1→3, and returns ``ndarray(5,)`` actions.
        """,
        "reset": """Reset all mutable state before a new episode (map seed).""",
    }
    body = docs.get(name)
    if body is None:
        body = f"Private helper `{name}` (see surrounding section comments in this module)."
    text = body.strip()
    lines = text.split("\n")
    out = [f'{pad}"""']
    out.extend(lines)
    out.append(f'{pad}"""')
    return "\n".join(out) + "\n"


def apply_renames(text: str) -> str:
    pairs = [
        ("_choose_detour_waypoint_connected_component", "_detour_waypoint_from_obstacle_cc"),
        ("_choose_clearest_direction_toward_center", "_clearest_horizontal_toward_search_center"),
        ("_get_front_patch_min_depth_m", "_min_depth_in_velocity_front_patch"),
        ("_obstacle_close_on_sides_not_front", "_lateral_close_obstacle_corridor_clear"),
        ("_center_depth_clear_at_least_m", "_is_forward_center_clear_depth"),
        ("_fallback_search_center_goal_detection", "_fallback_goal_toward_search_center"),
        ("_generate_random_point_in_search_area", "_sample_random_search_point"),
        ("_get_near_depth_average_direction", "_near_depth_bearing"),
        ("_goal_candidate_within_search_window", "_goal_in_search_bounds"),
        ("_update_detected_goal_candidate", "_merge_priority_goal_candidate"),
        ("_accel_scale_for_angular_velocity", "_angular_rate_accel_scale"),
        ("_speed_scale_for_approach_angle", "_approach_angle_speed_scale"),
        ("_pixel_to_direction_world", "_unproject_pixel_to_world_direction"),
        ("_direction_world_to_pixel", "_project_world_direction_to_pixel"),
        ("_depth_meters_for_direction", "_depth_along_world_direction"),
        ("_get_angular_velocity", "_read_angular_velocity"),
        ("_compute_search_area_center", "_world_search_center_from_offset"),
        ("_build_height_match_waypoints", "_waypoints_for_height_match"),
        ("_update_detected_goal_motion", "_update_goal_kinematics_ema"),
        ("_reset_detected_goal_motion", "_clear_goal_kinematics"),
        ("_tracked_goal_reference", "_lead_compensated_goal"),
        ("_state3_open_goal_action", "_state3_open_approach_command"),
        ("_goal_search_score", "_goal_candidate_distance_score"),
        ("_update_detected_goal", "_fuse_goal_observation"),
        ("_store_detected_goal", "_assign_detected_goal"),
        ("_front_obstacle_detected", "_front_patch_obstacle_too_close"),
        ("_infer_open_like_scene", "_depth_suggests_open_scene"),
        ("_state1_next_random_target", "_state1_next_search_waypoint"),
        ("_accel_scale_for_tilt", "_tilt_accel_scale"),
        ("_get_current_position", "_read_position_xyz"),
        ("_get_current_velocity", "_read_velocity_xyz"),
        ("_get_roll_pitch", "_read_roll_pitch_rad"),
        ("_get_look_direction_world", "_camera_forward_in_world"),
        ("_detect_goal_platform", "_find_goal_platform_from_depth"),
        ("_state1_move3d_kwargs", "_state1_motion_limits"),
        ("_get_search_area_vector", "_read_search_area_offset"),
        ("_get_yaw", "_read_yaw_rad"),
        ("_state3_goal_action", "_state3_straight_goal_approach_command"),
        ("_depth_median_m", "_median_scene_depth_m"),
    ]
    seen: set[str] = set()
    uniq = []
    for o, n in pairs:
        if o not in seen:
            seen.add(o)
            uniq.append((o, n))
    for old, new in sorted(uniq, key=lambda x: -len(x[0])):
        text = text.replace(f"def {old}(", f"def {new}(")
        text = text.replace(f"self.{old}(", f"self.{new}(")
    text = text.replace("def _move3d(", "def _velocity_command_toward_point(")
    text = text.replace("self._move3d(", "self._velocity_command_toward_point(")
    text = text.replace("def _get_front_pixel(", "def _movement_direction_pixel(")
    text = text.replace("self._get_front_pixel(", "self._movement_direction_pixel(")
    text = text.replace("def _batch_world_positions(", "def _batch_pixels_to_world_xyz(")
    text = text.replace("_batch_world_positions(", "_batch_pixels_to_world_xyz(")
    text = text.replace("def bfs_collect_component(", "def _bfs_edge_connected_component(")
    text = text.replace("bfs_collect_component(", "_bfs_edge_connected_component(")
    text = text.replace("def expand_component_with_similar_depth(", "def _grow_similar_depth_region(")
    text = text.replace("expand_component_with_similar_depth(", "_grow_similar_depth_region(")
    return text


def insert_rng_and_fix_init(text: str) -> str:
    if "RNG_SEED" not in text:
        text = text.replace(
            "    ELEVATED_SEARCH_HEIGHT_MARGIN_M = 2.0\n\n    def __init__(self):",
            "    ELEVATED_SEARCH_HEIGHT_MARGIN_M = 2.0\n"
            "    # Reproducible stochasticity for search randomization (benchmark-friendly).\n"
            "    RNG_SEED = 12345\n\n"
            "    def __init__(self):",
        )
    text = text.replace(
        "        self.detected_goal_velocity = None\n        self.detected_goal_velocity = None\n",
        "        self.detected_goal_velocity = None\n",
    )
    if "self.rng = np.random.default_rng" not in text.split("def __init__")[1].split("def ")[0]:
        text = text.replace(
            "        self.detected_goal_velocity = None\n\n    # ---------- Observation",
            "        self.detected_goal_velocity = None\n"
            "        self.rng = np.random.default_rng(self.RNG_SEED)\n\n"
            "    # ---------- Observation",
        )
    text = text.replace(
        "        offset_x = np.random.uniform(-radius_x, radius_x)\n"
        "        offset_y = np.random.uniform(-radius_y, radius_y)\n"
        "        offset_z = np.random.uniform(-radius_z, radius_z)\n",
        "        offset_x = self.rng.uniform(-radius_x, radius_x)\n"
        "        offset_y = self.rng.uniform(-radius_y, radius_y)\n"
        "        offset_z = self.rng.uniform(-radius_z, radius_z)\n",
    )
    text = text.replace(
        "            rng = np.random.RandomState(12345 + idx)\n",
        "            rng = np.random.default_rng(self.RNG_SEED + idx)\n",
    )
    text = text.replace(
        "turn_dir = self.state1_turn_direction if self.state1_turn_direction is not None else (1 if np.random.rand() < 0.5 else -1)\n",
        "turn_dir = self.state1_turn_direction if self.state1_turn_direction is not None else (1 if self.rng.random() < 0.5 else -1)\n",
    )
    text = text.replace(
        "        action = np.random.uniform(-1, 1, size=5)\n",
        "        action = self.rng.uniform(-1, 1, size=5)\n",
    )
    return text


def complete_reset(text: str) -> str:
    if "def reset(self):" not in text:
        return text
    tail = text.split("def reset(self):")[-1]
    if "self.detected_goal_velocity = None" in tail.split("def ")[0]:
        return text
    # Append missing fields before end of class - crude: replace incomplete reset block
    old = (
        "        self.detected_goal_priority = None\n"
        "        self.detected_goal_motion_ema = None\n"
    )
    new = (
        "        self.detected_goal_priority = None\n"
        "        self.detected_goal_motion_ema = None\n"
        "        self.detected_goal_velocity = None\n"
        "        self.rng = np.random.default_rng(self.RNG_SEED)\n"
    )
    if text.rstrip().endswith("self.detected_goal_motion_ema = None"):
        text = text.replace(
            "        self.detected_goal_priority = None\n"
            "        self.detected_goal_motion_ema = None\n",
            new,
            1,
        )
    return text


def insert_docstrings(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    inserts: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if _has_leading_docstring(node):
            continue
        if not node.body:
            continue
        name = node.name
        body_indent = int(node.col_offset) + 4
        doc = _doc_for_name(name, body_indent=body_indent)
        # Insert before first body statement (1-based lineno)
        li = node.body[0].lineno - 1
        inserts.append((li, doc))

    for li, doc in sorted(inserts, key=lambda x: -x[0]):
        lines.insert(li, doc)
    return "".join(lines)


def patch_module_header(text: str) -> str:
    header = '''"""
Swarm Subnet 124 — autonomous drone flight controller (miner submission).

Standalone module (NumPy/SciPy only). The simulator passes ``observation`` with
``state`` (ego) and ``depth`` (normalized image); :meth:`DroneFlightController.act`
returns a length-5 control vector each tick.

**Control states**

- **0 — Transit:** Fly staged waypoints toward the search volume.
- **1 — Search:** Random / ring exploration with obstacle detours and yaw search.
- **3 — Goal:** Approach, align, and descend onto the detected landing platform.

**Updates in this revision**

- Deterministic RNG via :attr:`RNG_SEED` and :attr:`DroneFlightController.rng` for all
  stochastic choices (search points, fallback actions).
- Private helpers renamed for readability (verb + object); public API unchanged
  (``act``, ``reset``, class name).

**Depth convention:** normalized ``d in [0,1]`` maps to metres as ``0.5 + 19.5 * d``.
"""

'''
    if text.startswith('"""Swarm Subnet 124'):
        text = re.sub(r'^"""[\s\S]*?"""\n\n', header, text, count=1)
    return text


def expand_class_docstring(text: str) -> str:
    old = '''class DroneFlightController:
    """
    Flight controller: state machine (0=move to search area, 1=search, 3=goal).
    All observation parsing, depth, movement, search area, and goal detection
    are inlined.
    """
'''
    new = '''class DroneFlightController:
    """
    Finite-state controller for the Swarm drone benchmark.

    State **0** moves into the search ellipsoid; **1** explores while avoiding
    obstacles; **3** tracks a fused depth-based goal pose. Goal geometry is inferred
    from depth edges and region statistics, not from privileged map data.

    Subcomponents are inlined (no external package imports): projection math,
    segmentation, PD-style velocity command generation, and goal fusion.
    """
'''
    if old in text:
        text = text.replace(old, new)
    return text


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    text = patch_module_header(text)
    text = expand_class_docstring(text)
    text = insert_rng_and_fix_init(text)
    text = apply_renames(text)
    text = complete_reset(text)
    text = insert_docstrings(text)
    PATH.write_text(text, encoding="utf-8")
    print(f"Wrote {PATH}")


if __name__ == "__main__":
    main()
