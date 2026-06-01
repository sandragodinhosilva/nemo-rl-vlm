import logging
import os
import re
from collections.abc import Iterable
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

_REMOTE_MEDIA_PREFIXES = ("http://", "https://", "data:")
_SELECTED_SESSIONS_RE = re.compile(
    r"^(?P<prefix>/mnt/data/sgsilva/tmp/[^/]+/selected_sessions/"
    r"(?P<split>train|validation|test))/(?P<session>[^/]+)(?P<suffix>/.*)?$"
)


def _is_local_media_path(value: Any) -> bool:
    return isinstance(value, str) and not value.startswith(_REMOTE_MEDIA_PREFIXES)


def resolve_local_media_path(media_path: str) -> str:
    """Resolve known stale THRIVE local media paths to their current location."""
    if not _is_local_media_path(media_path):
        return media_path

    match = _SELECTED_SESSIONS_RE.match(media_path)
    if match is None:
        return media_path

    if os.path.exists(media_path):
        return media_path

    suffix = match.group("suffix") or ""
    resolved_path = (
        f"/mnt/data/shared/vlm/data/10k/all/{match.group('session')}{suffix}"
    )
    if os.path.exists(resolved_path):
        return resolved_path

    return media_path


def normalize_thrive_media_paths(example: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a THRIVE example with local media paths normalized."""
    normalized = example.copy()

    for field_name in ("video_frames", "video", "image", "images"):
        value = normalized.get(field_name)
        if value is None:
            continue

        if isinstance(value, list):
            normalized[field_name] = [resolve_local_media_path(item) for item in value]
        elif isinstance(value, tuple):
            normalized[field_name] = tuple(
                resolve_local_media_path(item) for item in value
            )
        elif isinstance(value, str):
            normalized[field_name] = resolve_local_media_path(value)

    return normalized


def iter_local_media_paths(example: dict[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield local media paths stored on a THRIVE example."""
    normalized_example = normalize_thrive_media_paths(example)

    for field_name in ("video_frames", "video", "image", "images"):
        value = normalized_example.get(field_name)
        if value is None:
            continue

        if isinstance(value, (list, tuple)):
            for item in value:
                if _is_local_media_path(item):
                    yield field_name, item
        elif _is_local_media_path(value):
            yield field_name, value


def filter_dataset_missing_local_media(
    dataset,
    split_name: str,
    dataset_label: str,
    *,
    skip_missing_local_media_filter: bool = False,
):
    """Drop dataset rows that reference missing local image/video files."""
    total_examples = len(dataset)
    if total_examples == 0:
        return dataset

    if skip_missing_local_media_filter:
        logger.info(
            "Skipping eager local media existence filter for %s %s samples; "
            "normalizing paths only.",
            dataset_label,
            split_name,
        )
        return dataset.map(
            normalize_thrive_media_paths,
            desc=f"Normalizing {dataset_label} {split_name} media paths",
        )

    missing_paths_preview: list[str] = []

    def has_all_local_media(example: dict[str, Any]) -> bool:
        for field_name, media_path in iter_local_media_paths(example):
            if not os.path.exists(media_path):
                if len(missing_paths_preview) < 3:
                    missing_paths_preview.append(f"{field_name}={media_path}")
                return False
        return True

    filtered_dataset = dataset.filter(
        has_all_local_media,
        desc=f"Filtering {dataset_label} {split_name} samples with missing local media",
    )
    filtered_dataset = filtered_dataset.map(
        normalize_thrive_media_paths,
        desc=f"Normalizing {dataset_label} {split_name} media paths",
    )

    dropped_examples = total_examples - len(filtered_dataset)
    if dropped_examples > 0:
        preview = ", ".join(missing_paths_preview) if missing_paths_preview else "n/a"
        logger.warning(
            "Filtered %d/%d %s %s samples with missing local media. Examples: %s",
            dropped_examples,
            total_examples,
            dataset_label,
            split_name,
            preview,
        )

    return filtered_dataset


def load_rgb_image(image_path: str, example: dict[str, Any], field_name: str) -> Image.Image:
    """Load a local image path with a THRIVE-specific error message."""
    image_path = resolve_local_media_path(image_path)
    try:
        return Image.open(image_path).convert("RGB")
    except FileNotFoundError as exc:
        sample_bits = []
        for key in ("id", "sample_id", "uid", "dataset_type", "task_name"):
            value = example.get(key)
            if value is not None:
                sample_bits.append(f"{key}={value}")

        if not sample_bits and example.get("question"):
            sample_bits.append(f"question={str(example['question'])[:80]!r}")

        sample_desc = ", ".join(sample_bits) if sample_bits else "unidentified sample"
        raise FileNotFoundError(
            f"Missing THRIVE local media for {sample_desc} "
            f"(field '{field_name}'): {image_path}"
        ) from exc
