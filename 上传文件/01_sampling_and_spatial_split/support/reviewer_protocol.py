"""Auditable protocol helpers for reviewer-facing spatial experiments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


DISTANCE_SCALE_KEYS = (
    "meters_per_coordinate_unit",
    "metres_per_coordinate_unit",
    "m_per_coordinate_unit",
    "coordinate_unit_to_meters",
    "coordinate_unit_to_metres",
)
CELL_SIZE_KEYS = ("cell_size_m", "pixel_size_m", "grid_resolution_m")
COORDINATE_STEP_KEYS = (
    "coordinate_step",
    "coordinate_step_units",
    "grid_spacing_coordinate_units",
    "coordinate_grid_spacing",
)


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value):
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value.item() if isinstance(value, np.generic) else value


def _h5_attr_candidates(path: str) -> Dict[str, object]:
    values: Dict[str, object] = {}
    with h5py.File(path, "r") as handle:
        containers = [("root", handle)]
        if "metadata" in handle:
            containers.append(("metadata", handle["metadata"]))
        for prefix, container in containers:
            for key, value in container.attrs.items():
                values[f"{prefix}.{str(key)}"] = _scalar(value)
    return values


def resolve_distance_scale(
    h5_path: str,
    *,
    explicit_meters_per_coordinate_unit: Optional[float] = None,
    require_authoritative: bool = False,
) -> Dict[str, object]:
    """Resolve map-coordinate scale without inferring it from array dimensions."""
    if explicit_meters_per_coordinate_unit is not None:
        value = float(explicit_meters_per_coordinate_unit)
        if value <= 0 or not np.isfinite(value):
            raise ValueError("meters_per_coordinate_unit must be a finite value greater than zero.")
        return {
            "meters_per_coordinate_unit": value,
            "source": "explicit_user_input",
            "source_key": "--meters-per-coordinate-unit",
            "authoritative": True,
            "status": "resolved",
            "inference_used": False,
        }

    attrs = _h5_attr_candidates(h5_path)
    normalized = {key.lower().split(".")[-1]: (key, value) for key, value in attrs.items()}
    for candidate in DISTANCE_SCALE_KEYS:
        if candidate not in normalized:
            continue
        source_key, raw = normalized[candidate]
        value = float(raw)
        if value > 0 and np.isfinite(value):
            return {
                "meters_per_coordinate_unit": value,
                "source": "h5_metadata",
                "source_key": source_key,
                "authoritative": True,
                "status": "resolved",
                "inference_used": False,
            }

    cell_entry = next((normalized[key] for key in CELL_SIZE_KEYS if key in normalized), None)
    step_entry = next((normalized[key] for key in COORDINATE_STEP_KEYS if key in normalized), None)
    if cell_entry is not None and step_entry is not None:
        cell_source, cell_raw = cell_entry
        step_source, step_raw = step_entry
        cell_size_m = float(cell_raw)
        coordinate_step = float(step_raw)
        if cell_size_m > 0 and coordinate_step > 0 and np.isfinite(cell_size_m) and np.isfinite(coordinate_step):
            return {
                "meters_per_coordinate_unit": cell_size_m / coordinate_step,
                "source": "h5_cell_size_and_coordinate_step_metadata",
                "source_key": f"{cell_source}+{step_source}",
                "authoritative": True,
                "status": "resolved",
                "inference_used": False,
                "cell_size_m": cell_size_m,
                "coordinate_step": coordinate_step,
            }

    unresolved = {
        "meters_per_coordinate_unit": None,
        "source": "unresolved",
        "source_key": None,
        "authoritative": False,
        "status": "author_input_required",
        "inference_used": False,
        "message": (
            "The H5 file does not contain an authoritative metres-per-coordinate-unit scale. "
            "Supply --meters-per-coordinate-unit. A cell_size_m value alone is insufficient "
            "because map coordinates may be grid indices, metres, or another coordinate unit."
        ),
    }
    if require_authoritative:
        raise ValueError(unresolved["message"])
    return unresolved


def meters_to_coordinate_units(value_m: float, scale: Dict[str, object]) -> float:
    meters_per_unit = scale.get("meters_per_coordinate_unit")
    if meters_per_unit is None:
        raise ValueError("Distance conversion requested before meters_per_coordinate_unit was resolved.")
    return float(value_m) / float(meters_per_unit)


def distance_protocol(
    h5_path: str,
    *,
    explicit_meters_per_coordinate_unit: Optional[float],
    require_authoritative: bool,
    requested_meters: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    scale = resolve_distance_scale(
        h5_path,
        explicit_meters_per_coordinate_unit=explicit_meters_per_coordinate_unit,
        require_authoritative=require_authoritative,
    )
    requested = {str(key): float(value) for key, value in (requested_meters or {}).items()}
    converted = {}
    if scale.get("meters_per_coordinate_unit") is not None:
        converted = {key: meters_to_coordinate_units(value, scale) for key, value in requested.items()}
    return {
        **scale,
        "requested_distances_m": requested,
        "converted_distances_coordinate_units": converted,
        "coordinate_distance_unit": "input_map_coordinate_unit",
    }


def _read_xy(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    columns = {str(column).strip().lower(): column for column in frame.columns}
    x_col = next((columns[key] for key in ("x", "coord_x", "point_x", "east", "easting") if key in columns), None)
    y_col = next((columns[key] for key in ("y", "coord_y", "point_y", "north", "northing") if key in columns), None)
    if x_col is None or y_col is None:
        if frame.shape[1] < 2:
            raise KeyError(f"{path} must contain x/y columns.")
        x_col, y_col = frame.columns[:2]
    out = frame.copy()
    out["__x"] = pd.to_numeric(frame[x_col], errors="coerce")
    out["__y"] = pd.to_numeric(frame[y_col], errors="coerce")
    return out


def audit_expert_grouping(
    label_path: str,
    assignment_path: str,
    *,
    group_kind: str,
    expert_confirmed: bool,
    coordinate_tolerance: float = 1e-6,
) -> Dict[str, object]:
    """Audit assignment coverage and provenance; never manufacture expert groups."""
    if not expert_confirmed:
        raise ValueError(f"{group_kind} grouping must be explicitly confirmed by a geological expert.")
    labels = _read_xy(label_path)
    assignment = _read_xy(assignment_path)
    group_aliases = {
        "camp": ("camp_id", "camp", "ore_camp", "矿田", "矿田编号"),
        "fault": ("fault_id", "fault", "fid", "断裂", "断裂编号"),
    }
    aliases = group_aliases.get(str(group_kind).lower())
    if aliases is None:
        raise ValueError(f"Unknown grouping kind: {group_kind}")
    colmap = {str(column).strip().lower(): column for column in assignment.columns}
    group_col = next((colmap[key] for key in aliases if key in colmap), None)
    if group_col is None:
        raise KeyError(f"Assignment CSV needs one of: {', '.join(aliases)}")

    group_text = assignment[group_col].astype(str).str.strip()
    missing_group = group_text.str.lower().isin({"", "nan", "none", "null", "-1"})
    valid_assignment = assignment.loc[
        assignment[["__x", "__y"]].notna().all(axis=1) & (~missing_group)
    ].copy()
    valid_assignment["__group"] = group_text.loc[valid_assignment.index]
    rounded_assignment = set(
        zip(
            np.round(valid_assignment["__x"].to_numpy(dtype=float) / max(coordinate_tolerance, 1e-12)).astype(np.int64),
            np.round(valid_assignment["__y"].to_numpy(dtype=float) / max(coordinate_tolerance, 1e-12)).astype(np.int64),
        )
    )
    label_xy = labels[["__x", "__y"]].dropna().to_numpy(dtype=float)
    rounded_labels = [
        (
            int(round(float(x) / max(coordinate_tolerance, 1e-12))),
            int(round(float(y) / max(coordinate_tolerance, 1e-12))),
        )
        for x, y in label_xy
    ]
    unmatched = [index for index, key in enumerate(rounded_labels) if key not in rounded_assignment]
    duplicate_xy_count = int(valid_assignment.duplicated(subset=["__x", "__y"], keep=False).sum())
    duplicate_deposit_ids = 0
    deposit_col = next((colmap[key] for key in ("deposit_id", "mineral_id", "unit_id") if key in colmap), None)
    if deposit_col is not None:
        duplicate_deposit_ids = int(valid_assignment[deposit_col].duplicated(keep=False).sum())
    counts = valid_assignment.groupby("__group", dropna=False).size().sort_index()
    errors = []
    if len(valid_assignment) == 0:
        errors.append("assignment_has_no_valid_rows")
    if int(len(counts)) < 2:
        errors.append("fewer_than_two_groups")
    if unmatched:
        errors.append(
            "label_coordinates_missing_from_assignment"
            f"(unmatched={len(unmatched)}/{len(rounded_labels)}; "
            "矿点文件与矿田表坐标必须逐点一致，请使用对齐版归属表)"
        )
    if duplicate_deposit_ids:
        errors.append("duplicate_deposit_ids")
    if missing_group.any():
        errors.append("missing_group_ids")
    payload = {
        "group_kind": str(group_kind),
        "status": "valid" if not errors else "invalid",
        "expert_confirmed": True,
        "assignment_path": os.path.abspath(assignment_path),
        "assignment_sha256": sha256_file(assignment_path),
        "label_path": os.path.abspath(label_path),
        "label_sha256": sha256_file(label_path),
        "label_count": int(len(label_xy)),
        "assignment_valid_count": int(len(valid_assignment)),
        "group_count": int(len(counts)),
        "group_sizes": {str(key): int(value) for key, value in counts.items()},
        "unmatched_label_count": int(len(unmatched)),
        "unmatched_label_row_indices": unmatched[:100],
        "missing_group_count": int(missing_group.sum()),
        "duplicate_coordinate_row_count": duplicate_xy_count,
        "duplicate_deposit_id_row_count": duplicate_deposit_ids,
        "coordinate_tolerance": float(coordinate_tolerance),
        "errors": errors,
    }
    if errors:
        raise ValueError(f"Invalid expert {group_kind} grouping: {', '.join(errors)}")
    return payload


def merge_mineralization_units(
    frame: pd.DataFrame,
    distance_coordinate_units: float,
    *,
    protected_group_columns: Sequence[str] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Deterministically merge connected deposits within a distance threshold."""
    data = frame.reset_index(drop=True).copy()
    threshold = float(distance_coordinate_units)
    data["source_deposit_index"] = np.arange(len(data), dtype=np.int64)
    if threshold <= 0 or len(data) <= 1:
        data["unit_id"] = np.arange(len(data), dtype=np.int64)
        mapping = data[["source_deposit_index", "unit_id", "x", "y"]].copy()
        return data, mapping, {
            "enabled": False,
            "threshold_coordinate_units": threshold,
            "source_deposit_count": int(len(data)),
            "merged_unit_count": int(len(data)),
            "cross_group_edges_blocked": 0,
        }

    coords = data[["x", "y"]].to_numpy(dtype=float)
    parent = np.arange(len(data), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    protected = [column for column in protected_group_columns if column in data.columns]
    blocked = 0
    try:
        from scipy.spatial import cKDTree

        pairs: Iterable[Tuple[int, int]] = sorted(cKDTree(coords).query_pairs(threshold))
    except Exception:
        pairs = (
            (left, right)
            for left in range(len(data))
            for right in range(left + 1, len(data))
            if float(np.linalg.norm(coords[left] - coords[right])) <= threshold
        )
    for left, right in pairs:
        if protected and any(str(data.loc[left, col]) != str(data.loc[right, col]) for col in protected):
            blocked += 1
            continue
        union(int(left), int(right))

    roots = [find(index) for index in range(len(data))]
    ordered_roots = {root: unit_id for unit_id, root in enumerate(sorted(set(roots)))}
    data["unit_id"] = [ordered_roots[root] for root in roots]
    rows = []
    for unit_id, members in data.groupby("unit_id", sort=True):
        row = members.iloc[0].copy()
        row["x"] = float(members["x"].mean())
        row["y"] = float(members["y"].mean())
        row["unit_id"] = int(unit_id)
        row["source_deposit_count"] = int(len(members))
        row["source_deposit_indices"] = ";".join(str(int(value)) for value in members["source_deposit_index"])
        rows.append(row)
    merged = pd.DataFrame(rows).reset_index(drop=True)
    mapping_columns = ["source_deposit_index", "unit_id", "x", "y", *protected]
    mapping = data[mapping_columns].copy()
    return merged, mapping, {
        "enabled": True,
        "threshold_coordinate_units": threshold,
        "source_deposit_count": int(len(data)),
        "merged_unit_count": int(len(merged)),
        "merged_deposit_count": int(len(data) - len(merged)),
        "protected_group_columns": protected,
        "cross_group_edges_blocked": int(blocked),
    }


def environment_manifest(*, random_seeds: Optional[Sequence[int]] = None) -> Dict[str, object]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "torch", "h5py", "shap"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = "not_available"
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "random_seeds": [int(value) for value in (random_seeds or [])],
    }


def data_manifest(paths: Dict[str, str]) -> Dict[str, object]:
    records = []
    for role, raw_path in paths.items():
        path = str(raw_path or "").strip()
        if not path:
            continue
        record = {"role": str(role), "path": os.path.abspath(path), "exists": os.path.exists(path)}
        if os.path.isfile(path):
            record.update({"size_bytes": int(os.path.getsize(path)), "sha256": sha256_file(path)})
        records.append(record)
    return {"files": records}


def write_json(path: str, payload: Dict[str, object]) -> str:
    abs_path = os.path.abspath(os.path.normpath(path))

    def _long(p: str) -> str:
        if os.name != "nt" or p.startswith("\\\\?\\"):
            return p
        if p.startswith("\\\\"):
            return "\\\\?\\UNC\\" + p.lstrip("\\")
        return "\\\\?\\" + p

    parent = os.path.dirname(abs_path) or "."
    try:
        os.makedirs(parent, exist_ok=True)
        open_path = abs_path if len(abs_path) < 240 else _long(abs_path)
        with open(open_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    except OSError:
        if os.name != "nt":
            raise
        long_path = _long(abs_path)
        os.makedirs(os.path.dirname(long_path), exist_ok=True)
        with open(long_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return path
