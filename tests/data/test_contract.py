from __future__ import annotations

import torch

from src.data.contract import JOINT_ORDER, WindowSample


def test_window_sample_exposes_complete_optional_geometry_contract() -> None:
    sample = WindowSample(
        hamer_tokens=torch.zeros(1, 2, 2, 1024),
        hamer_valid=torch.zeros(1, 2, 2, dtype=torch.bool),
        vggt_tokens=torch.zeros(1, 2, 17, 2048),
        vggt_valid=torch.zeros(1, 2, dtype=torch.bool),
        keypoints=torch.zeros(1, 2, 2, 21, 3),
        bbox=torch.zeros(1, 2, 2, 4),
        bbox_valid=torch.zeros(1, 2, 2, dtype=torch.bool),
        camera_extrinsics=torch.eye(4).expand(1, 2, 4, 4),
        camera_intrinsics=torch.eye(3).expand(1, 2, 3, 3),
        camera_valid=torch.ones(1, 2, dtype=torch.bool),
        hand_joints3d=torch.zeros(1, 2, 2, 16, 3),
        joint_valid=torch.ones(1, 2, 2, 16, dtype=torch.bool),
    )
    assert len(JOINT_ORDER) == 16
    assert sample.camera_extrinsics.shape == (1, 2, 4, 4)
    assert sample.camera_intrinsics.shape == (1, 2, 3, 3)
    assert sample.hand_joints3d.shape == (1, 2, 2, 16, 3)
