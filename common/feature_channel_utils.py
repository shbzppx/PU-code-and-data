from __future__ import annotations

import ast
from typing import Iterable, List, Optional, Sequence

import h5py
import numpy as np


def _decode_text_list(values) -> List[str]:
    result: List[str] = []
    if values is None:
        return result
    if isinstance(values, (list, tuple, np.ndarray)):
        iterable = values.tolist() if isinstance(values, np.ndarray) else values
        for item in iterable:
            if isinstance(item, bytes):
                result.append(item.decode("utf-8", errors="ignore"))
            else:
                result.append(str(item))
        return result
    if isinstance(values, bytes):
        return [values.decode("utf-8", errors="ignore")]
    return [str(values)]


def parse_selected_channels(value) -> Optional[List[int]]:
    if value in (None, "", [], (), set()):
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        parsed = [int(part) for part in parts if part]
        return parsed or None
    if isinstance(value, Iterable):
        parsed = [int(item) for item in value]
        return parsed or None
    return None


def serialize_selected_channels(selected_channels: Optional[Sequence[int]]) -> str:
    values = parse_selected_channels(selected_channels)
    if not values:
        return ""
    return ",".join(str(int(index)) for index in values)


def infer_h5_channel_names(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as handle:
        file_names = _decode_text_list(handle.attrs.get("file_names"))
        source_type = str(handle.attrs.get("source_type", "") or "").strip().lower()
        metadata_attrs = dict(handle["metadata"].attrs) if "metadata" in handle else {}

        channel_count = None
        if "fused_features" in handle:
            fused = handle["fused_features"]
            if fused.ndim >= 3:
                channel_count = int(fused.shape[0])
        elif "windows" in handle:
            windows = handle["windows"]
            if windows.ndim == 4:
                channel_count = int(windows.shape[2])
            elif windows.ndim == 3:
                channel_count = 1
        elif "samples" in handle:
            samples = handle["samples"]
            if samples.ndim == 4:
                channel_count = int(samples.shape[1])
            elif samples.ndim == 3:
                channel_count = 1
            elif samples.ndim == 2:
                channel_count = int(samples.shape[1]) if samples.shape[1] > 1 else 1
        elif "vectors" in handle:
            vectors = handle["vectors"]
            if vectors.ndim == 2:
                channel_count = int(vectors.shape[1])
            elif vectors.ndim == 1:
                channel_count = 1

        if channel_count is None:
            try:
                channel_count = int(metadata_attrs.get("channels", 0) or 0)
            except (TypeError, ValueError):
                channel_count = 0

        if channel_count <= 0:
            return []

        if len(file_names) == channel_count:
            return [str(name) for name in file_names]

        source_files = metadata_attrs.get("source_files")
        try:
            parsed_source_files = ast.literal_eval(str(source_files)) if source_files not in (None, "") else []
        except (SyntaxError, ValueError):
            parsed_source_files = []
        source_file_names = [str(item).split("\\")[-1].split("/")[-1] for item in parsed_source_files]
        if len(source_file_names) == channel_count:
            return source_file_names

        prefix = "图层"
        if source_type == "grid_fused_features":
            prefix = "特征图层"
        return [f"{prefix}{index + 1}" for index in range(channel_count)]


def normalize_selected_channels(selected_channels, channel_count: int) -> Optional[List[int]]:
    parsed = parse_selected_channels(selected_channels)
    if not parsed:
        return None
    normalized = sorted({int(index) for index in parsed if 0 <= int(index) < int(channel_count)})
    if not normalized or len(normalized) >= int(channel_count):
        return None
    return normalized


def describe_selected_channels(selected_channels, channel_names: Sequence[str], *, max_names: int = 6) -> str:
    channel_names = list(channel_names or [])
    total = len(channel_names)
    if total == 0:
        return "图层: 未识别"
    normalized = normalize_selected_channels(selected_channels, total)
    if normalized is None:
        return f"图层: 全选 ({total}/{total})"
    chosen_names = [channel_names[index] for index in normalized if 0 <= index < total]
    preview = "、".join(chosen_names[:max_names])
    if len(chosen_names) > max_names:
        preview += "..."
    return f"图层: 已选 {len(chosen_names)}/{total} ({preview})"


def subset_samples_by_channels(
    samples: np.ndarray,
    metadata: Optional[dict],
    selected_channels,
) -> tuple[np.ndarray, dict]:
    samples = np.asarray(samples, dtype=np.float32)
    metadata = dict(metadata or {})
    channel_count = int(samples.shape[1]) if samples.ndim >= 2 else 0
    available_channel_names = metadata.get("available_channel_names")
    if not available_channel_names:
        available_channel_names = [f"图层{index + 1}" for index in range(channel_count)]
    available_channel_names = list(available_channel_names)

    normalized = normalize_selected_channels(selected_channels, channel_count)
    if normalized is None:
        metadata["available_channel_names"] = available_channel_names
        metadata["selected_channel_indices"] = list(range(channel_count))
        metadata["selected_channel_names"] = list(available_channel_names)
        metadata["channel_selection_active"] = False
        metadata["input_channels"] = int(channel_count)
        return samples, metadata

    subset = np.asarray(samples[:, normalized, ...], dtype=np.float32)
    metadata["available_channel_names"] = available_channel_names
    metadata["selected_channel_indices"] = list(normalized)
    metadata["selected_channel_names"] = [available_channel_names[index] for index in normalized]
    metadata["channel_selection_active"] = True
    metadata["input_channels"] = int(subset.shape[1])
    return subset, metadata
