"""Target-area reporting with manual blind/extension labels (OGR R1.7)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _deposit_coords(df: pd.DataFrame) -> np.ndarray:
    if df is None or len(df) == 0:
        return np.empty((0, 2), dtype=np.float64)
    cols = {str(c).strip().lower(): c for c in df.columns}
    x_col = next((cols[k] for k in ("x", "east", "easting") if k in cols), df.columns[0])
    y_col = next((cols[k] for k in ("y", "north", "northing") if k in cols), df.columns[1])
    return df[[x_col, y_col]].to_numpy(dtype=np.float64)


def summarize_targets_from_mask(
    binary_mask: np.ndarray,
    *,
    coordinate_cell_size: Optional[float] = None,
    meters_per_coordinate_unit: Optional[float] = None,
    cell_size_m: Optional[float] = None,
    known_deposits: Optional[pd.DataFrame] = None,
    supporting_layers: Optional[Sequence[str]] = None,
    uncertainty_map: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Connected-component targets from a boolean prospectivity mask."""
    from scipy import ndimage

    mask = np.asarray(binary_mask, dtype=bool)
    labeled, n_labels = ndimage.label(mask)
    deposits = _deposit_coords(known_deposits) if known_deposits is not None else np.empty((0, 2))
    rows: List[Dict[str, object]] = []
    height, width = mask.shape
    meta = metadata or {}
    if cell_size_m is not None:
        if coordinate_cell_size is not None or meters_per_coordinate_unit is not None:
            raise ValueError("Use either cell_size_m or coordinate_cell_size + meters_per_coordinate_unit, not both.")
        coordinate_cell_size = 1.0
        meters_per_coordinate_unit = float(cell_size_m)
    if coordinate_cell_size is None or meters_per_coordinate_unit is None:
        raise ValueError(
            "Target reporting requires coordinate_cell_size and meters_per_coordinate_unit; "
            "no 50 m fallback is permitted."
        )
    coordinate_cell_size = float(coordinate_cell_size)
    meters_per_coordinate_unit = float(meters_per_coordinate_unit)
    if coordinate_cell_size <= 0 or meters_per_coordinate_unit <= 0:
        raise ValueError("Distance scale values must be greater than zero.")
    physical_cell_size_m = coordinate_cell_size * meters_per_coordinate_unit
    x_min = float(meta.get("x_min", 0.0))
    y_max = float(meta.get("y_max", height * coordinate_cell_size))
    x_max = float(meta.get("x_max", width * coordinate_cell_size))
    y_min = float(meta.get("y_min", 0.0))

    def _xy(row: int, col: int):
        x = x_min + col / max(width - 1, 1) * (x_max - x_min)
        y = y_max - row / max(height - 1, 1) * (y_max - y_min)
        return float(x), float(y)

    for lab in range(1, n_labels + 1):
        ys, xs = np.where(labeled == lab)
        if len(xs) == 0:
            continue
        area_km2 = float(len(xs) * (physical_cell_size_m ** 2) / 1e6)
        cy, cx = float(np.mean(ys)), float(np.mean(xs))
        center_x, center_y = _xy(int(round(cy)), int(round(cx)))
        if len(deposits):
            d = np.sqrt((deposits[:, 0] - center_x) ** 2 + (deposits[:, 1] - center_y) ** 2)
            min_dist = float(np.min(d)) * meters_per_coordinate_unit
        else:
            min_dist = float("nan")
        unc_mean = float("nan")
        if uncertainty_map is not None:
            unc = np.asarray(uncertainty_map, dtype=np.float64)
            unc_mean = float(np.nanmean(unc[labeled == lab]))
        rows.append(
            {
                "target_id": f"T{lab:02d}",
                "cell_count": int(len(xs)),
                "area_km2": area_km2,
                "center_x": center_x,
                "center_y": center_y,
                "distance_to_nearest_known_deposit_m": min_dist,
                "mean_uncertainty": unc_mean,
                "supporting_layers": ";".join(supporting_layers or []),
                "target_type": "",  # manual: blind / extension
                "geological_rationale": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def merge_manual_target_annotations(
    auto_df: pd.DataFrame,
    manual_csv: str,
) -> pd.DataFrame:
    """Merge manual blind/extension labels from CSV with columns target_id,target_type,..."""
    manual = pd.read_csv(manual_csv)
    if "target_id" not in manual.columns:
        raise KeyError("manual target CSV must include target_id")
    merged = auto_df.merge(manual, on="target_id", how="left", suffixes=("", "_manual"))
    for col in ("target_type", "geological_rationale", "notes"):
        manual_col = f"{col}_manual"
        if manual_col in merged.columns:
            merged[col] = merged[manual_col].where(merged[manual_col].notna(), merged.get(col))
            merged = merged.drop(columns=[manual_col])
    return merged


def export_target_report(
    output_dir: str,
    binary_mask: np.ndarray,
    *,
    coordinate_cell_size: Optional[float] = None,
    meters_per_coordinate_unit: Optional[float] = None,
    cell_size_m: Optional[float] = None,
    known_deposits: Optional[pd.DataFrame] = None,
    supporting_layers: Optional[Sequence[str]] = None,
    uncertainty_map: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, object]] = None,
    manual_annotation_csv: Optional[str] = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    auto = summarize_targets_from_mask(
        binary_mask,
        coordinate_cell_size=coordinate_cell_size,
        meters_per_coordinate_unit=meters_per_coordinate_unit,
        cell_size_m=cell_size_m,
        known_deposits=known_deposits,
        supporting_layers=supporting_layers,
        uncertainty_map=uncertainty_map,
        metadata=metadata,
    )
    template_path = os.path.join(output_dir, "targets_manual_annotation_template.csv")
    auto[["target_id", "target_type", "geological_rationale", "notes"]].to_csv(
        template_path, index=False, encoding="utf-8-sig"
    )
    if manual_annotation_csv and os.path.exists(manual_annotation_csv):
        final = merge_manual_target_annotations(auto, manual_annotation_csv)
    else:
        final = auto
    summary_path = os.path.join(output_dir, "targets_summary.csv")
    final.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta_path = os.path.join(output_dir, "targets_summary_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_targets": int(len(final)),
                "manual_annotation_csv": manual_annotation_csv,
                "template_csv": template_path,
                "note": "Fill target_type as blind or extension manually, then re-run merge.",
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return {"summary_csv": summary_path, "template_csv": template_path, "meta_json": meta_path}
