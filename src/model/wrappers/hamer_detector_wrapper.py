"""Official HaMeR front-end detector used by the handbox cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from detectron2.config import LazyConfig
from detectron2.data import transforms as T


class HamerHandBoxDetector:
    """Run ViTDet person detection followed by ViTPose hand boxes."""

    def __init__(
        self,
        *,
        repository: str | Path,
        detector_checkpoint: str | Path,
        device: str = "cuda",
        image_size: int = 384,
        precision: str = "fp16",
    ) -> None:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = True
        self.repository = Path(repository).resolve()
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.precision = precision
        if str(self.repository) not in sys.path:
            sys.path.insert(0, str(self.repository))
        import hamer
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        cfg_path = self.repository / "hamer" / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        cfg = LazyConfig.load(str(cfg_path))
        cfg.train.init_checkpoint = str(Path(detector_checkpoint).resolve())
        cfg.model.backbone.square_pad = 0
        cfg.dataloader.test.mapper.augmentations[0].short_edge_length = self.image_size
        cfg.dataloader.test.mapper.augmentations[0].max_size = self.image_size
        for index in range(3):
            cfg.model.roi_heads.box_predictors[index].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(cfg)
        if precision == "fp16":
            self.detector.model.half()
        elif precision != "fp32":
            raise ValueError("precision must be fp32 or fp16")
        previous_cwd = os.getcwd()
        try:
            os.chdir(self.repository)
            self.pose = ViTPoseModel(self.device)
        finally:
            os.chdir(previous_cwd)

    def _prepare(self, image: Tensor) -> dict:
        array = image.numpy()
        bgr = np.ascontiguousarray(array[:, :, ::-1])
        height, width = bgr.shape[:2]
        augmented = T.AugInput(bgr)
        self.detector.aug(augmented)
        dtype = np.float16 if self.precision == "fp16" else np.float32
        tensor = torch.as_tensor(augmented.image.astype(dtype).transpose(2, 0, 1))
        return {"image": tensor, "height": height, "width": width}

    @torch.inference_mode()
    def forward(self, images: Tensor) -> list[dict]:
        if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.uint8:
            raise ValueError("detector images must be uint8 [N,3,H,W]")
        inputs = [self._prepare(image.permute(1, 2, 0).contiguous()) for image in images]
        detector_outputs = self.detector.model(inputs)
        predictions = []
        for image, output in zip(images, detector_outputs):
            instances = output["instances"]
            keep = (instances.pred_classes == 0) & (instances.scores > 0.5)
            person_boxes = instances.pred_boxes.tensor[keep].detach().cpu().numpy().astype(np.float32)
            person_scores = instances.scores[keep].detach().cpu().numpy().astype(np.float32)
            order = np.argsort(-person_scores)
            person_boxes = person_boxes[order]
            person_scores = person_scores[order]
            poses = []
            if len(person_boxes):
                person_input = np.concatenate([person_boxes, person_scores[:, None]], axis=1)
                poses = self.pose.predict_pose(image.permute(1, 2, 0).numpy(), [person_input])
            selected = poses[0] if poses else None
            hand_keypoints = np.zeros((2, 21, 3), dtype=np.float32)
            boxes = np.zeros((2, 4), dtype=np.float32)
            valid = np.zeros(2, dtype=bool)
            hand_scores = np.zeros(2, dtype=np.float32)
            if selected is not None:
                for hand_index, keypoint_slice in enumerate((slice(-42, -21), slice(-21, None))):
                    keypoints = selected["keypoints"][keypoint_slice].astype(np.float32)
                    if keypoints.shape != (21, 3):
                        continue
                    keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=0.0, neginf=0.0)
                    hand_keypoints[hand_index] = keypoints
                    good = keypoints[:, 2] > 0.5
                    if int(good.sum()) > 3:
                        box = np.array([
                            keypoints[good, 0].min(), keypoints[good, 1].min(),
                            keypoints[good, 0].max(), keypoints[good, 1].max(),
                        ], dtype=np.float32)
                        if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                            boxes[hand_index] = box
                            valid[hand_index] = True
                            hand_scores[hand_index] = float(keypoints[good, 2].mean())
            predictions.append({
                "boxes": torch.from_numpy(boxes),
                "box_valid": torch.from_numpy(valid),
                "valid": torch.from_numpy(valid),
                "keypoints": torch.from_numpy(hand_keypoints),
                "metadata": {
                    "person_boxes": person_boxes.tolist(),
                    "person_scores": person_scores.tolist(),
                    "hand_scores": hand_scores.tolist(),
                    "coordinate_space": "rectified_upright_rgb",
                    "image_size": self.image_size,
                    "mode": f"vitdet_h_{self.precision}_{self.image_size}_no_square_pad",
                },
            })

        return predictions

    __call__ = forward
