import numpy as np
import pytest

from src.data.hot3d import Hot3DRectifier
from src.data.hot3d.dataset import Hot3DDataset


@pytest.fixture
def calibration():
    return pytest.importorskip("projectaria_tools.core.calibration")


def test_box_and_image_share_clockwise_rotation(calibration):
    rectifier = Hot3DRectifier(output_size=32)
    camera = calibration.get_linear_camera_calibration(32, 32, 20, "test")
    raw = np.zeros((32, 32, 3), dtype=np.uint8)
    raw[7:17, 3:10] = 255
    image = rectifier.rectify_image(raw, camera, camera)
    box = rectifier.rectify_box([3, 7, 9, 16], camera, camera)
    np.testing.assert_array_equal(box, [15, 3, 24, 9])
    ys, xs = np.nonzero(image[..., 0] > 127)
    np.testing.assert_array_equal(box, [xs.min(), ys.min(), xs.max(), ys.max()])


def test_partial_box_is_clipped_and_invisible_box_is_none(calibration):
    rectifier = Hot3DRectifier(output_size=32)
    camera = calibration.get_linear_camera_calibration(32, 32, 20, "test")
    np.testing.assert_array_equal(
        rectifier.rectify_box([-10, 7, 9, 16], camera, camera), [15, 0, 24, 9]
    )
    for box in [[-10, 7, -1, 16], [32, 0, 40, 8], [0, 32, 10, 50], [0, -20, 10, -1], [3, 4, 3, 8], [0, 0, np.nan, 8]]:
        assert rectifier.rectify_box(box, camera, camera) is None


@pytest.mark.parametrize("clip,frame", [
    ("002477", 22), ("002477", 30), ("002477", 31),
    ("002892", 82), ("002892", 83),
])
def test_real_fisheye_extrapolation_does_not_create_full_image_box(calibration, clip, frame):
    dataset = Hot3DDataset(clip, [frame], Hot3DRectifier(output_size=384))
    assert dataset.hand_box("left", frame) is None
    box = dataset.hand_box("right", frame)
    assert box is not None and np.isfinite(box).all()
    assert (box >= 0).all() and (box <= 383).all()


def test_real_partially_visible_box_matches_warped_image_support(calibration):
    rectifier = Hot3DRectifier(output_size=384)
    dataset = Hot3DDataset("001849", [5], rectifier)
    source, destination, _ = dataset._calibration(5)
    raw_box = dataset.hands(5)["left"]["boxes_amodal"]["214-1"]
    width, height = source.get_image_size()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, raw_box)
    image[y1:y2 + 1, x1:x2 + 1] = 255
    upright = rectifier.rectify_image(image, source, destination)
    ys, xs = np.nonzero(upright[..., 0] > 127)
    actual = dataset.hand_box("left", 5)
    np.testing.assert_allclose(actual, [xs.min(), ys.min(), xs.max(), ys.max()], atol=1)
    assert actual[3] == 383
