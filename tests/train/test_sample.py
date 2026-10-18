from pathlib import Path

import torch
from torch import nn

import src.train.sample as sample
from src.model.flow.state import CameraFlowState, ManoFlowState


class _Rectifier:
    output_size = 384

    def pca_components(self, hand):
        return torch.zeros(15, 45), torch.zeros(45)

    def mano_joints(self, hand, root, transl, local):
        return torch.zeros(root.shape[0], 16, 3)


class _Dataset:
    def __init__(self, clip, frames, rectifier, **kwargs):
        self.clip_id = clip
        self.frames = list(frames)

    def image(self, frame):
        return torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()

    def hands(self, frame):
        pose = {"wrist_xform": [0.0] * 6, "thetas": [0.0] * 15}
        return {"left": {"mano_pose": pose}, "right": {"mano_pose": pose}}

    def camera_pose(self, frame):
        return {
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "translation_xyz": [0.0, 0.0, 0.0],
        }

    def _calibration(self, frame):
        return None, None, torch.eye(3).numpy()

    def targets(self, rectifier):
        return (
            ManoFlowState(torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 6), torch.zeros(1, 2, 2, 15, 6)),
            CameraFlowState(torch.zeros(1, 2, 6), torch.zeros(1, 2, 3)),
        )


class _HandBox:
    cache_root = Path("/tmp")

    def __init__(self):
        self.called = False

    def get_boxes(self, dataset):
        self.called = True
        boxes = torch.ones(2, 2, 4)
        valid = torch.tensor([[True, False], [True, True]])
        return boxes, valid, []


class _Hamer:
    cache_root = Path("/tmp")
    def __init__(self):
        self.called = None

    def get_tokens(self, dataset, boxes, valid):
        self.called = (boxes.clone(), valid.clone())
        return torch.zeros(2, 2, 1024), valid.clone()


class _VGGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(0))

    def forward(self, images):
        return torch.zeros(1, images.shape[1], 17, 2048)


def test_sample_window_uses_predicted_boxes_not_dataset_hand_box(monkeypatch):
    monkeypatch.setattr(sample, "frame_ids", lambda clip: [0, 1])
    monkeypatch.setattr(sample, "Hot3DDataset", _Dataset)
    handbox = _HandBox()
    hamer = _Hamer()
    result = sample.sample_window(
        clip_ids=["000001"],
        window=2,
        hamer_cache=hamer,
        handbox_cache=handbox,
        vggt=_VGGT(),
        rectifier=_Rectifier(),
        seed=0,
    )
    assert handbox.called
    assert hamer.called is not None
    torch.testing.assert_close(result.bbox[0], hamer.called[0] / 384.0)
    assert torch.equal(result.bbox_valid[0], hamer.called[1])
