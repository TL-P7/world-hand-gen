from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from src.data.egodex import EgoDexDataset, JOINT_SUFFIXES, episode_ids


@pytest.fixture
def episode(tmp_path):
    episode_id = "test/pick/0"
    directory = tmp_path / "test" / "pick"
    directory.mkdir(parents=True)
    (directory / "0.mp4").touch()
    with h5py.File(directory / "0.hdf5", "w") as annotation:
        for hi, side in enumerate(("left", "right")):
            for ji, suffix in enumerate(JOINT_SUFFIXES):
                transform = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
                transform[:, :3, 3] = np.array([
                    [100 * hi + ji, frame, -ji - 1] for frame in range(4)
                ])
                annotation.create_dataset(f"transforms/{side}{suffix}", data=transform)
    return tmp_path, episode_id


def test_official_splits_and_pairing(episode):
    root, _ = episode
    for part in range(1, 6):
        directory = root / f"part{part}" / "pick"
        directory.mkdir(parents=True)
        (directory / "0.hdf5").touch()
        (directory / "0.mp4").touch()
    assert episode_ids("test", root) == ["test/pick/0"]
    assert episode_ids("train", root) == [f"part{part}/pick/0" for part in range(1, 6)]
    (root / "test/pick/1.mp4").touch()
    with pytest.raises(ValueError, match="unpaired"):
        episode_ids("test", root)
    with pytest.raises(ValueError, match="split"):
        episode_ids("part1", root)


def test_raw_op16_order_and_window(episode):
    root, episode_id = episode
    assert JOINT_SUFFIXES == (
        "Hand", "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip",
        "IndexFingerKnuckle", "IndexFingerIntermediateBase", "IndexFingerIntermediateTip",
        "MiddleFingerKnuckle", "MiddleFingerIntermediateBase", "MiddleFingerIntermediateTip",
        "RingFingerKnuckle", "RingFingerIntermediateBase", "RingFingerIntermediateTip",
        "LittleFingerKnuckle", "LittleFingerIntermediateBase", "LittleFingerIntermediateTip",
    )
    dataset = EgoDexDataset(episode_id, [1, 2], root)
    assert dataset._images == {} and dataset._joints is None
    xyz, valid = dataset.joints()
    assert xyz.shape == (2, 2, 16, 3)
    assert xyz.dtype == np.float32
    assert valid.shape == (2, 2, 16) and valid.dtype == bool
    assert valid.all()
    np.testing.assert_array_equal(xyz[0, :, :, 0], np.tile(np.arange(16), (2, 1)))
    np.testing.assert_array_equal(xyz[1, :, :, 0], np.tile(100 + np.arange(16), (2, 1)))
    np.testing.assert_array_equal(xyz[:, 0, :, 1], np.ones((2, 16)))
    np.testing.assert_array_equal(xyz[:, 1, :, 1], np.full((2, 16), 2))
    np.testing.assert_array_equal(xyz[0, 0, :, 2], -np.arange(16) - 1)
    assert dataset.joints() is dataset._joints


def test_optional_confidence_and_missing_points(episode):
    root, episode_id = episode
    with h5py.File(root / "test/pick/0.hdf5", "r+") as annotation:
        annotation.create_dataset("confidences/leftHand", data=[1, 0, np.nan, -1])
        annotation.create_dataset("confidences/rightHand", data=[0.2, 0.3, 1, 0.4])
        annotation["transforms/rightThumbKnuckle"][2, 0, 3] = np.nan
        del annotation["transforms/leftLittleFingerIntermediateTip"]
    xyz, valid = EgoDexDataset(episode_id, root=root).joints()
    np.testing.assert_array_equal(valid[0, :, 0], [True, False, False, False])
    assert valid[1, :, 0].all()
    assert not valid[1, 2, 1]
    assert np.isnan(xyz[1, 2, 1, 0])
    assert np.isnan(xyz[0, :, 15]).all()
    assert not valid[0, :, 15].any()
    assert valid[0, :, 1].all()


def test_missing_hand(episode):
    root, episode_id = episode
    with h5py.File(root / "test/pick/0.hdf5", "r+") as annotation:
        for suffix in JOINT_SUFFIXES:
            del annotation[f"transforms/right{suffix}"]
    xyz, valid = EgoDexDataset(episode_id, root=root).joints()
    assert valid[0].all()
    assert not valid[1].any()
    assert np.isnan(xyz[1]).all()


@pytest.mark.parametrize("frames", [[], [-1], [4], [0, 2], [1, 0], [1.0], [True]])
def test_invalid_frames(episode, frames):
    root, episode_id = episode
    with pytest.raises(ValueError, match="frames"):
        EgoDexDataset(episode_id, frames, root)


def test_bad_annotation_shapes(episode):
    root, episode_id = episode
    with h5py.File(root / "test/pick/0.hdf5", "r+") as annotation:
        annotation.create_dataset("confidences/leftHand", data=[1])
    with pytest.raises(ValueError, match="confidence shape"):
        EgoDexDataset(episode_id, root=root).joints()
    with h5py.File(root / "test/pick/0.hdf5", "r+") as annotation:
        del annotation["transforms/leftHand"]
        annotation.create_dataset("transforms/leftHand", data=np.zeros((4, 3)))
    with pytest.raises(ValueError, match="transform shape"):
        EgoDexDataset(episode_id, root=root)


def test_missing_pair_and_invalid_episode_id(episode):
    root, episode_id = episode
    for invalid in ("/test/pick/0", "../pick/0", "pick/0"):
        with pytest.raises(ValueError, match="episode_id"):
            EgoDexDataset(invalid, root=root)
    (root / "test/pick/0.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        EgoDexDataset(episode_id, root=root)


def test_rgb_video_read_and_frame_alignment(episode):
    cv2 = pytest.importorskip("cv2")

    root, episode_id = episode
    path = str(root / "test/pick/0.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
    assert writer.isOpened()
    try:
        for frame in range(4):
            image = np.full((24, 32, 3), (frame * 30, 0, 200), dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()
    dataset = EgoDexDataset(episode_id, [1, 2], root)
    rgb = dataset.image(2)
    assert rgb.shape == (24, 32, 3) and rgb.dtype == np.uint8
    np.testing.assert_allclose(rgb.mean(axis=(0, 1)), [200, 0, 60], atol=8)
    assert dataset.image(2) is rgb
    np.testing.assert_allclose(dataset.image(1).mean(axis=(0, 1)), [200, 0, 30], atol=8)
    with pytest.raises(ValueError, match="outside"):
        dataset.image(0)


def test_video_frame_count_mismatch(episode):
    cv2 = pytest.importorskip("cv2")

    root, episode_id = episode
    path = str(root / "test/pick/0.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 24))
    assert writer.isOpened()
    try:
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    with pytest.raises(ValueError, match="frame mismatch"):
        EgoDexDataset(episode_id, root=root).image(0)
