#!/usr/bin/env python3
"""Convert goal_detector.pt to goal_detector.onnx."""

import torch
import numpy as np
from pathlib import Path

# Import the model class from the original code
from drone_agent_pt_backup import EndPlatformBinaryLocalizerNet


def convert(pt_path: str = "goal_detector.pt", onnx_path: str = "goal_detector.onnx"):
    pt_path = Path(pt_path)
    onnx_path = Path(onnx_path)

    device = torch.device("cpu")
    payload = torch.load(pt_path, map_location=device, weights_only=False)

    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    dropout = float(config.get("dropout", 0.10))

    model = EndPlatformBinaryLocalizerNet(in_channels=3, aux_dim=1, dropout=dropout)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    # The model only defines forward_raw; alias it to forward for ONNX export
    model.forward = model.forward_raw
    model.eval()

    # Save aux stats and config alongside ONNX for the runtime to load
    np.savez(
        onnx_path.with_suffix(".npz"),
        aux_mean=np.asarray(payload["aux_mean"], dtype=np.float32),
        aux_std=np.asarray(payload["aux_std"], dtype=np.float32),
        goal_visibility_threshold=np.asarray(
            [float(config.get("goal_visibility_threshold", 0.5))], dtype=np.float32
        ),
    )

    # Dummy inputs matching actual depth image size (128x128)
    dummy_image = torch.randn(1, 3, 128, 128)
    dummy_aux = torch.randn(1, 1)

    # Use dynamo=False for the legacy export path (compatible with torch 2.10+)
    torch.onnx.export(
        model,
        (dummy_image, dummy_aux),
        str(onnx_path),
        input_names=["image", "aux"],
        output_names=["vis_logits", "heatmap_logits", "z_map"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "aux": {0: "batch"},
            "vis_logits": {0: "batch"},
            "heatmap_logits": {0: "batch", 2: "height", 3: "width"},
            "z_map": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Exported aux stats to {onnx_path.with_suffix('.npz')}")


if __name__ == "__main__":
    convert()
