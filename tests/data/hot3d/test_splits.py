from __future__ import annotations

from src.data.hot3d.splits import partition_clips


def test_partition_matches_requested_counts_and_is_deterministic() -> None:
    clips = [f"{index:06d}" for index in range(1516)]
    train, test = partition_clips(clips, train_count=1272, seed=2026)
    repeated_train, repeated_test = partition_clips(
        list(reversed(clips)), train_count=1272, seed=2026
    )
    assert len(train) == 1272
    assert len(test) == 244
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(clips)
    assert train == repeated_train
    assert test == repeated_test
