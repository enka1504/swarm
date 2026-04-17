#!/usr/bin/env python3
from __future__ import annotations

import torch
from typing import Optional
import math

from pathlib import Path

import numpy as np
from torch import nn
from torch.nn import functional as F

MAX_YAW_RATE = 3.141
SIM_DT = 1 / 50
CAMERA_FOV_DEG = 90.0
CAMERA_OFFSET_M = 0.13
CAMERA_UP_OFFSET_M = 0.05


def _normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"expected a 3D vector, got shape {arr.shape}")

    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        raise ValueError("cannot normalize a near-zero vector")
    return (arr / norm).astype(np.float32)

def rotation_matrix_from_roll_pitch_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Right-handed rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    vector_world = R @ vector_body
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation_x = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ], dtype=np.float32)

    rotation_y = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ], dtype=np.float32)

    rotation_z = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return (rotation_z @ rotation_y @ rotation_x).astype(np.float32)

def _transpose_observation_images(observation: dict) -> dict:
    if "depth" not in observation or "state" not in observation:
        raise KeyError("Observation missing 'depth' or 'state' keys.")

    # Process depth
    # ----------------------------------
    depth = np.asarray(observation["depth"], dtype=np.float32).copy()
    if depth.ndim != 3:
        raise ValueError(f"Expected depth observation with 3 dimensions (H, W, C), got shape {depth.shape}.")

    # Convert to channels-first
    depth = np.transpose(depth, (2, 0, 1))

    return {
        "depth": depth,
        "state": observation["state"]
    }


def _camera_geometry_from_pose(
    drone_position: np.ndarray,
    drone_rpy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rotation_matrix = rotation_matrix_from_roll_pitch_yaw(drone_rpy[0], drone_rpy[1], drone_rpy[2])
    forward_direction = _normalize_vector(rotation_matrix @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up_guess = _normalize_vector(rotation_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right_direction = np.cross(forward_direction, up_guess).astype(np.float32)

    if float(np.linalg.norm(right_direction)) <= 1e-8:
        right_direction = np.cross(forward_direction, np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)

    right_direction = _normalize_vector(right_direction)
    up_direction = _normalize_vector(np.cross(right_direction, forward_direction))

    camera_position = (
        np.asarray(drone_position, dtype=np.float32)
        + forward_direction * CAMERA_OFFSET_M
        + up_guess * CAMERA_UP_OFFSET_M
    ).astype(np.float32)
    camera_target = (camera_position + forward_direction * 20.0).astype(np.float32)

    return camera_position, camera_target, right_direction, up_direction, forward_direction


def _uvz_to_world(
    uvz: np.ndarray,
    camera_position: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
    camera_forward: np.ndarray,
    fov_deg: float,
) -> np.ndarray:
    uvz_arr = np.asarray(uvz, dtype=np.float32)
    z_cam = max(float(uvz_arr[2]), 1e-3)
    half_tan = float(np.tan(np.deg2rad(np.float32(fov_deg)) * 0.5))
    x_cam = float(uvz_arr[0]) * z_cam * half_tan
    y_cam = float(uvz_arr[1]) * z_cam * half_tan

    return (
        np.asarray(camera_position, dtype=np.float32)
        + np.asarray(camera_right, dtype=np.float32) * x_cam
        + np.asarray(camera_up, dtype=np.float32) * y_cam
        + np.asarray(camera_forward, dtype=np.float32) * z_cam
    ).astype(np.float32)


def _build_binary_localizer_input(depth: np.ndarray) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 3 or depth_arr.shape[0] != 1:
        raise ValueError(f"expected depth shape (1,H,W), got {depth_arr.shape}")

    _, height, width = depth_arr.shape
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(1.0, -1.0, height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)
    x_grid = x_grid[None, :, :].astype(np.float32, copy=False)
    y_grid = y_grid[None, :, :].astype(np.float32, copy=False)

    return np.concatenate([depth_arr, x_grid, y_grid], axis=0).astype(np.float32, copy=False)


def _build_binary_localizer_aux(fov_deg: float) -> np.ndarray:
    return np.asarray([fov_deg], dtype=np.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = max(1, out_channels // 16)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EndPlatformBinaryLocalizerNet(nn.Module):
    """Architecture used by RL/train_goal_detector.py."""

    def __init__(self, in_channels: int = 3, aux_dim: int = 1, dropout: float = 0.10):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32, stride=1)
        self.enc2 = ConvBlock(32, 64, stride=2)
        self.enc3 = ConvBlock(64, 128, stride=2)
        self.enc4 = ConvBlock(128, 192, stride=2)
        self.bottleneck = ConvBlock(192, 256, stride=2)

        self.aux_proj = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )

        self.dec4 = ConvBlock(256 + 64 + 192, 192, stride=1)
        self.dec3 = ConvBlock(192 + 128, 128, stride=1)
        self.dec2 = ConvBlock(128 + 64, 64, stride=1)
        self.dec1 = ConvBlock(64 + 32, 32, stride=1)

        self.dropout = nn.Dropout2d(p=dropout)
        self.heatmap_head = nn.Conv2d(32, 1, kernel_size=1)
        self.z_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward_raw(self, image: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(image)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        aux_feat = self.aux_proj(aux)
        aux_map = aux_feat.unsqueeze(-1).unsqueeze(-1)
        aux_map = aux_map.expand(-1, -1, b.shape[2], b.shape[3])
        d4_in = F.interpolate(torch.cat([b, aux_map], dim=1), scale_factor=2, mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4_in, e4], dim=1))
        d3_in = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3_in, e3], dim=1))
        d2_in = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2_in, e2], dim=1))
        d1_in = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1_in, e1], dim=1))
        d1 = self.dropout(d1)
        heatmap_logits = self.heatmap_head(d1)
        vis_logits = _visibility_logits_from_heatmap_logits(heatmap_logits)
        return vis_logits, heatmap_logits, self.z_head(d1)


def _visibility_logits_from_heatmap_logits(heatmap_logits: torch.Tensor) -> torch.Tensor:
    flat_logits = heatmap_logits.reshape(heatmap_logits.shape[0], -1)
    detection_energy = torch.logsumexp(flat_logits, dim=1, keepdim=True) - math.log(flat_logits.shape[1])
    return detection_energy

def slerp_direction(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Normalize inputs
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return b

    a /= na
    b /= nb

    # Angle and axis
    cross = np.cross(a, b)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arctan2(cross_norm, dot)  # in [0, π]

    if angle < 1e-8:  # almost the same direction
        v = b
    elif np.pi - angle < 1e-8:  # opposite: pick a stable perpendicular axis
        x = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, x)) > 0.9:
            x = np.array([0.0, 1.0, 0.0])
        k = np.cross(a, x)
        k /= np.linalg.norm(k)
        c, s = np.cos(0.03 * angle), np.sin(0.03 * angle)
        v = a * c + np.cross(k, a) * s + k * (np.dot(k, a)) * (1 - c)
    else:
        k = cross / cross_norm
        c, s = np.cos(0.03 * angle), np.sin(0.03 * angle)
        v = a * c + np.cross(k, a) * s + k * (np.dot(k, a)) * (1 - c)

    return v / np.linalg.norm(v)


class DroneFlightController:
    def __init__(
        self,
        *,
        goal_detector_model_path: Optional[Path] = None,
    ):
        script_dir = Path(__file__).resolve().parent

        if goal_detector_model_path is None:
            goal_detector_model_path = script_dir / "goal_detector.pt"

        self._detect_device()
        self._load_goal_detector_model(goal_detector_model_path)

        self._mode = "takeoff"
        self._last_action = None
        self._landing_platform_position = None

    def act(self, observation):
        state = observation.get("state", None).squeeze()

        drone_position = np.array([state[0], state[1], state[2]], dtype=float)
        drone_rpy = np.array([state[3], state[4], state[5]], dtype=float)
        drone_velocity = np.array([state[6], state[7], state[8]], dtype=float)
        drone_speed = float(np.linalg.norm(drone_velocity))
        drone_altitude = state[-4] * 20.0

        search_area_vector = np.array([state[-3], state[-2], state[-1]], dtype=float)
        search_area_position = search_area_vector + drone_position

        search_area_position[2] += 2.5
        search_area_vector = search_area_position - drone_position
        distance_to_search_area = float(np.linalg.norm(search_area_vector))

        processed_observation = _transpose_observation_images(observation)
        processed_depth = processed_observation["depth"]

        goal_visibility_prob, predicted_goal_position = self._predict_goal_visibility_and_position(
            processed_depth,
            drone_position,
            drone_rpy,
        )
        is_goal_visible = bool(goal_visibility_prob >= self._goal_visibility_threshold)
        visible_goal_position = None

        if is_goal_visible:
            visible_goal_position = predicted_goal_position.copy()

        if self._mode == "takeoff":
            yaw_to_search_area = np.arctan2(search_area_vector[1], search_area_vector[0])
            yaw_to_search_area_diff = (yaw_to_search_area - drone_rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
            yaw_command = yaw_to_search_area / np.pi
            z_command = 0.0
            speed_command = 0.3
            min_altitude = 1.5

            if drone_altitude < min_altitude:
                z_command = 1.0

            if drone_altitude >= min_altitude and is_goal_visible:
                self._mode = "navigation"
            elif drone_altitude >= min_altitude and abs(yaw_to_search_area_diff) < (np.pi / 36):
                self._mode = "search"

            action = np.array([0.0, 0.0, z_command, speed_command, yaw_command], dtype=np.float32)
        elif self._mode == "search":
            acceleration_rate = 0.05
            brake_rate = 0.025
            drone_speed_normalized = min(drone_speed, 3.0) / 3.0

            if self._landing_platform_position is None:
                if distance_to_search_area > 3.0:
                    yaw_command = np.arctan2(search_area_vector[1], search_area_vector[0]) / np.pi
                    speed_command = min(drone_speed_normalized + acceleration_rate, 1.0)
                else:
                    yaw_command = self._infinite_rotation_to_left(drone_rpy[2])
                    speed_command = distance_to_search_area / 3.0

                    if speed_command < self._last_action[3] - brake_rate:
                        speed_command = self._last_action[3] - brake_rate

                    speed_command = max(speed_command, 0.0)

                direction = search_area_vector

                if float(np.linalg.norm(direction)) > 1e-6:
                    direction_norm = direction / np.linalg.norm(direction)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                action = np.concatenate([direction_norm, [speed_command, yaw_command]], dtype=np.float32)
            else:
                direction = self._last_action[0:3]
                speed_command = self._last_action[3] - brake_rate
                speed_command = max(speed_command, 0.0)
                yaw_command = self._infinite_rotation_to_left(drone_rpy[2])

                action = np.concatenate([direction, [speed_command, yaw_command]], dtype=np.float32)

            if is_goal_visible:
                self._mode = "navigation"
        elif self._mode == "navigation":
            acceleration_rate = 0.05
            drone_speed_normalized = min(drone_speed, 3.0) / 3.0
            speed_command = min(drone_speed_normalized + acceleration_rate, 1.0)

            if visible_goal_position is not None:
                goal_position = visible_goal_position.copy()
                goal_position[2] += 0.5

                direction = goal_position - drone_position
                distance_to_goal = float(np.linalg.norm(direction))
                yaw_command = np.arctan2(direction[1], direction[0]) / np.pi

                if distance_to_goal > 1e-6:
                    direction[2] *= 5.0
                    direction_norm = direction / np.linalg.norm(direction)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                if drone_altitude < 0.5:
                    direction_norm = np.array([direction_norm[0] * 0.1, direction_norm[1] * 0.1, 1.0], dtype=np.float32)

                if distance_to_goal < 4.0:
                    self._mode = "landing"
                    self._landing_platform_position = None

                action = np.concatenate([direction_norm, [speed_command, yaw_command]], dtype=np.float32)
            else:
                action = self._last_action.copy()

            if not is_goal_visible or visible_goal_position is None:
                self._mode = "search"
        elif self._mode == "landing":
            if visible_goal_position is not None:
                goal_position = visible_goal_position.copy()
                goal_position[2] += 0.3

                if self._landing_platform_position is None:
                    self._landing_platform_position = goal_position

                direction_to_landing_point = self._landing_platform_position - drone_position
                direction_to_current_goal_position = goal_position - drone_position
                distance_to_landing_point = float(np.linalg.norm(direction_to_landing_point))
                horizontal_distance_to_current_goal_position = float(
                    np.linalg.norm(goal_position[0:2] - drone_position[0:2]))

                brake_rate = 0.01
                speed_command = self._last_action[3] - brake_rate
                speed_command = max(speed_command, 0.1)

                yaw_command = np.arctan2(direction_to_current_goal_position[1],
                                         direction_to_current_goal_position[0]) / np.pi

                if distance_to_landing_point > 0.2:
                    direction_to_landing_point[2] *= 2.0
                    direction_norm = direction_to_landing_point / np.linalg.norm(direction_to_landing_point)
                else:
                    direction_norm = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    speed_command = 0.0

                if horizontal_distance_to_current_goal_position < 0.6:
                    direction_norm = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                    speed_command = 0.3

                action = np.concatenate([direction_norm, [speed_command, yaw_command]], dtype=np.float32)
            else:
                action = self._last_action.copy()

            if not is_goal_visible or visible_goal_position is None:
                self._mode = "search"
        else:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if self._last_action is not None:
            action = np.concatenate([slerp_direction(self._last_action[0:3], action[0:3]), [action[3], action[4]]]).astype(np.float32)

        action = np.clip(action, -1.0, 1.0)

        self._last_action = action

        return action[None, :]

    def reset(self):
        self._mode = "takeoff"
        self._last_action = None
        self._landing_platform_position = None

    def _infinite_rotation_to_left(self, drone_yaw):
        max_yaw_change = MAX_YAW_RATE * SIM_DT
        new_drone_yaw_angle = drone_yaw + max_yaw_change - 1e-4
        new_drone_yaw_angle_normalized = new_drone_yaw_angle / np.pi

        if new_drone_yaw_angle_normalized > 1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized - 1.0) - 1.0

        if new_drone_yaw_angle_normalized < -1.0:
            new_drone_yaw_angle_normalized = (new_drone_yaw_angle_normalized + 1.0) + 1.0

        return np.clip(new_drone_yaw_angle_normalized, -1.0, 1.0)

    def _load_goal_detector_model(self, goal_detector_model_path: Path):
        if not goal_detector_model_path.exists():
            raise FileNotFoundError(f"Model not found: {goal_detector_model_path}")

        payload = torch.load(goal_detector_model_path, map_location=self._device, weights_only=False)
        if "model_state_dict" not in payload:
            raise KeyError(f"Model missing model_state_dict: {goal_detector_model_path}")
        if "aux_mean" not in payload or "aux_std" not in payload:
            raise KeyError("Unsupported binary-localizer checkpoint format.")

        feature_layout = payload.get("feature_layout") if isinstance(payload.get("feature_layout"), dict) else {}
        if feature_layout.get("image_channels") != ["depth_normalized_0_1", "x_coord_grid", "y_coord_grid"]:
            raise KeyError("Unsupported binary-localizer checkpoint format.")

        aux_mean = np.asarray(payload["aux_mean"], dtype=np.float32)
        aux_std = np.asarray(payload["aux_std"], dtype=np.float32)
        if aux_mean.shape != (1,) or aux_std.shape != (1,):
            raise ValueError(
                "Binary-localizer aux stats shape mismatch: "
                f"mean={aux_mean.shape}, std={aux_std.shape}"
            )

        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        dropout = float(config.get("dropout", 0.10))
        model = EndPlatformBinaryLocalizerNet(
            in_channels=3,
            aux_dim=1,
            dropout=dropout,
        ).to(self._device)
        load_result = model.load_state_dict(payload["model_state_dict"], strict=False)
        unexpected = set(load_result.unexpected_keys)
        missing = set(load_result.missing_keys)
        allowed_legacy = {
            "visibility_head.0.weight",
            "visibility_head.0.bias",
            "visibility_head.1.weight",
            "visibility_head.1.bias",
            "visibility_head.4.weight",
            "visibility_head.4.bias",
        }
        if missing:
            raise RuntimeError(
                "Binary-localizer checkpoint is missing required parameters: "
                f"{sorted(missing)}"
            )
        if unexpected - allowed_legacy:
            raise RuntimeError(
                "Binary-localizer checkpoint has unexpected parameters: "
                f"{sorted(unexpected - allowed_legacy)}"
            )
        model.eval()

        self._goal_detector_model = model
        self._goal_detector_aux = (aux_mean, aux_std)
        self._goal_visibility_threshold = float(config.get("goal_visibility_threshold", 0.5))

    @torch.no_grad()
    def _predict_goal_visibility_and_position(
        self,
        depth: np.ndarray,
        drone_position: np.ndarray,
        drone_rpy: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        camera_position, _, camera_right, camera_up, camera_forward = _camera_geometry_from_pose(
            drone_position,
            drone_rpy,
        )

        aux_mean, aux_std = self._goal_detector_aux
        aux_features = _build_binary_localizer_aux(CAMERA_FOV_DEG)
        aux_features = (aux_features - aux_mean) / aux_std

        image_input = _build_binary_localizer_input(np.asarray(depth, dtype=np.float32))
        x_image = torch.from_numpy(image_input[None, :]).to(device=self._device, dtype=torch.float32)
        x_aux = torch.from_numpy(aux_features[None, :]).to(device=self._device, dtype=torch.float32)
        vis_logits_t, heatmap_logits_t, z_map_t = self._goal_detector_model.forward_raw(x_image, x_aux)
        visibility_prob, pred_uvz = self._decode_binary_localizer_prediction(vis_logits_t, heatmap_logits_t, z_map_t)

        pred_world = _uvz_to_world(
            pred_uvz,
            camera_position,
            camera_right,
            camera_up,
            camera_forward,
            CAMERA_FOV_DEG,
        )
        return visibility_prob, pred_world

    def _decode_binary_localizer_prediction(
        self,
        vis_logits_t: torch.Tensor,
        heatmap_logits_t: torch.Tensor,
        z_map_t: torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        _, _, height, width = heatmap_logits_t.shape
        y_pix_t, x_pix_t = torch.meshgrid(
            torch.arange(height, device=self._device, dtype=torch.float32),
            torch.arange(width, device=self._device, dtype=torch.float32),
            indexing="ij",
        )
        u_grid_t = (2.0 * x_pix_t / float(width - 1) - 1.0).reshape(-1)
        v_grid_t = (1.0 - 2.0 * y_pix_t / float(height - 1)).reshape(-1)

        probs_t = torch.softmax(heatmap_logits_t.reshape(1, -1), dim=1).squeeze(0)
        z_pos_t = F.softplus(z_map_t.reshape(-1)) + 1e-3

        pred_u = torch.sum(probs_t * u_grid_t)
        pred_v = torch.sum(probs_t * v_grid_t)
        pred_z = torch.sum(probs_t * z_pos_t)

        visibility_prob = float(torch.sigmoid(vis_logits_t).squeeze().detach().cpu().item())
        pred_uvz = np.asarray(
            [float(pred_u.detach().cpu().item()), float(pred_v.detach().cpu().item()), float(pred_z.detach().cpu().item())],
            dtype=np.float32,
        )
        return visibility_prob, pred_uvz

    def _detect_device(self):
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
