import logging

import pytest
from datasets import Dataset
from PIL import Image

from nemo_rl.data.datasets.response_datasets.thrive_vlm import (
    format_thrive_vlm_dataset,
)
from nemo_rl.data.datasets.response_datasets.thrive_vlm_common import (
    filter_dataset_missing_local_media,
    load_rgb_image,
    normalize_thrive_media_paths,
    resolve_local_media_path,
)


def test_format_thrive_vlm_dataset_keeps_paths_when_return_pil_false():
    example = {
        "image": ["/tmp/does-not-need-to-exist.png"],
        "question": "What is shown?",
        "answer": "A test image.",
    }

    formatted = format_thrive_vlm_dataset(example, return_pil=False)

    assert formatted["messages"][0]["content"][0]["image"] == example["image"][0]


def test_filter_dataset_missing_local_media_drops_missing_rows(tmp_path, caplog):
    valid_path = tmp_path / "valid.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(valid_path)
    missing_path = tmp_path / "missing.png"

    dataset = Dataset.from_dict(
        {
            "image": [str(valid_path), str(missing_path)],
            "question": ["valid", "missing"],
            "answer": ["yes", "no"],
        }
    )

    with caplog.at_level(logging.WARNING):
        filtered = filter_dataset_missing_local_media(
            dataset, "train", "THRIVE-VLM SFT"
        )

    assert len(filtered) == 1
    assert filtered[0]["question"] == "valid"
    assert "missing local media" in caplog.text


def test_resolve_local_media_path_remaps_selected_sessions_path(tmp_path, monkeypatch):
    session_id = "10001_123456_01012025123456_55555555"
    stale = (
        "/mnt/data/sgsilva/tmp/mcqa_2403_rebuild_20260330/selected_sessions/train/"
        f"{session_id}/images/frame.webp"
    )
    fresh = tmp_path / "10k" / "all" / session_id / "images" / "frame.webp"
    fresh.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(fresh)

    original_exists = resolve_local_media_path.__globals__["os"].path.exists

    def fake_exists(path):
        if path == str(fresh):
            return True
        if path == stale:
            return False
        return original_exists(path)

    monkeypatch.setattr(
        "nemo_rl.data.datasets.response_datasets.thrive_vlm_common.os.path.exists",
        fake_exists,
    )
    monkeypatch.setattr(
        "nemo_rl.data.datasets.response_datasets.thrive_vlm_common._SELECTED_SESSIONS_RE",
        resolve_local_media_path.__globals__["re"].compile(
            r"^(?P<prefix>/mnt/data/sgsilva/tmp/[^/]+/selected_sessions/"
            r"(?P<split>train|validation|test))/(?P<session>[^/]+)(?P<suffix>/.*)?$"
        ),
    )

    expected = f"/mnt/data/shared/vlm/data/10k/all/{session_id}/images/frame.webp"
    assert resolve_local_media_path(stale) == expected


def test_normalize_thrive_media_paths_updates_video_and_frames(tmp_path, monkeypatch):
    session_id = "10001_123456_01012025123456_55555555"
    example = {
        "video": (
            "/mnt/data/sgsilva/tmp/mcqa_2403_rebuild_20260330/selected_sessions/train/"
            f"{session_id}"
        ),
        "video_frames": [
            "/mnt/data/sgsilva/tmp/mcqa_2403_rebuild_20260330/selected_sessions/train/"
            f"{session_id}/images/frame.webp"
        ],
    }

    def fake_exists(path):
        return path.startswith(f"/mnt/data/shared/vlm/data/10k/all/{session_id}")

    monkeypatch.setattr(
        "nemo_rl.data.datasets.response_datasets.thrive_vlm_common.os.path.exists",
        fake_exists,
    )

    normalized = normalize_thrive_media_paths(example)
    assert normalized["video"] == f"/mnt/data/shared/vlm/data/10k/all/{session_id}"
    assert normalized["video_frames"][0] == (
        f"/mnt/data/shared/vlm/data/10k/all/{session_id}/images/frame.webp"
    )


def test_load_rgb_image_raises_contextual_error():
    with pytest.raises(FileNotFoundError, match="id=sample-123"):
        load_rgb_image(
            "/tmp/this-file-does-not-exist.png",
            {"id": "sample-123", "task_name": "thrive-vlm"},
            "image",
        )
