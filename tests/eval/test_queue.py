import csv
import json
from pathlib import Path

from src.eval.protocol import PROTOCOL
from src.eval.queue import _completed, _output_dir


def test_completed_requires_exact_checkpoint_and_full_artifacts(tmp_path):
    checkpoint = tmp_path / "source" / "step_00002000.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    output = tmp_path / "eval" / "step_00002000"
    output.mkdir(parents=True)
    (output / "request.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "checkpoint_step": 2000,
    }))
    (output / "summary.json").write_text(json.dumps({
        "checkpoint_step": 2000, "split": "test244", "clips": 244,
        "windows": 488, "smoke_only": False,
        "metrics": {str(i): 0.0 for i in range(25)},
    }))
    (output / "predictions.pt").write_bytes(b"artifact")
    with (output / "records.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip", "window"])
        writer.writerows(("001849", i) for i in range(488))
    assert _completed(output, checkpoint, 2000, protocol=False)
    assert not _completed(output, checkpoint, 2000)
    summary = json.loads((output / "summary.json").read_text())
    summary.update({"protocol": PROTOCOL, "filter_offscreen": True,
                    "hand_windows": 424, "excluded_windows": 64})
    (output / "summary.json").write_text(json.dumps(summary))
    assert _completed(output, checkpoint, 2000)
    other = tmp_path / "other" / "step_00002000.pt"
    assert not _completed(output, other, 2000)
    (output / "predictions.pt").unlink()
    assert not _completed(output, checkpoint, 2000)


def test_retry_never_overwrites_failed_output(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    assert _output_dir(eval_dir, 14000) == eval_dir / "step_00014000"
    (eval_dir / "step_00014000").mkdir()
    assert _output_dir(eval_dir, 14000) == eval_dir / "step_00014000_retry_01"
    (eval_dir / "step_00014000_retry_01").mkdir()
    assert _output_dir(eval_dir, 14000) == eval_dir / "step_00014000_retry_02"
