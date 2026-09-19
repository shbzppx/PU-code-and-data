"""OGR revision spatial validation splits: leave-one-camp / leave-one-fault / variogram-block."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _ensure_frame(minerals: Optional[pd.DataFrame]) -> pd.DataFrame:
    if minerals is None:
        return pd.DataFrame(columns=["x", "y"])
    frame = minerals.reset_index(drop=True).copy()
    if "x" not in frame.columns or "y" not in frame.columns:
        raise KeyError("Mineral frame must contain x/y columns.")
    return frame


def _normalize_column_map(frame: pd.DataFrame) -> Dict[str, object]:
    return {str(c).strip().lower(): c for c in frame.columns}


def load_deposit_camp_assignment(path: str) -> pd.DataFrame:
    """Load expert ore-camp assignment CSV (x,y,camp_id [deposit_id] [basin_id])."""
    frame = pd.read_csv(path)
    column_map = _normalize_column_map(frame)
    x_col = next((column_map[k] for k in ("x", "coord_x", "east", "easting") if k in column_map), None)
    y_col = next((column_map[k] for k in ("y", "coord_y", "north", "northing") if k in column_map), None)
    camp_col = next(
        (
            column_map[k]
            for k in ("camp_id", "camp", "ore_camp", "矿田", "矿田编号", "cluster_id")
            if k in column_map
        ),
        None,
    )
    if x_col is None or y_col is None or camp_col is None:
        raise KeyError("deposit_camp_assignment CSV needs x, y, camp_id columns.")
    out = pd.DataFrame(
        {
            "x": pd.to_numeric(frame[x_col], errors="coerce"),
            "y": pd.to_numeric(frame[y_col], errors="coerce"),
            "camp_id": pd.to_numeric(frame[camp_col], errors="coerce"),
        }
    )
    if "deposit_id" in column_map:
        out["deposit_id"] = frame[column_map["deposit_id"]]
    else:
        out["deposit_id"] = np.arange(len(out))
    basin_col = next(
        (
            column_map[k]
            for k in ("basin_id", "basin", "catchment_id", "watershed_id", "汇水盆地", "汇水")
            if k in column_map
        ),
        None,
    )
    if basin_col is not None:
        out["basin_id"] = pd.to_numeric(frame[basin_col], errors="coerce")
    out = out.dropna(subset=["x", "y", "camp_id"]).reset_index(drop=True)
    out["camp_id"] = out["camp_id"].astype(np.int64)
    return out


def read_basin_grd(path: str) -> dict:
    """Read Surfer DSAA basin/catchment grid (.grd)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    if not lines or not lines[0].strip().upper().startswith("DSAA"):
        raise ValueError(f"Not a DSAA GRD file: {path}")
    nx, ny = map(int, lines[1].split())
    xmin, xmax = map(float, lines[2].split())
    ymin, ymax = map(float, lines[3].split())
    vals: List[float] = []
    for line in lines[5:]:
        vals.extend(float(v) for v in line.split())
    arr = np.asarray(vals[: nx * ny], dtype=np.float64)
    grid = arr.reshape(ny, nx)
    grid = np.where(np.abs(grid) > 1e30, np.nan, grid)
    return {
        "Z": grid,
        "nx": int(nx),
        "ny": int(ny),
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "path": str(path),
    }


def sample_basin_ids_at_coords(basin_grid: dict, coords: np.ndarray) -> np.ndarray:
    """Nearest-cell sampling of basin IDs (DSAA row0 = ymin)."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        return np.full(0, np.nan, dtype=np.float64)
    z = np.asarray(basin_grid["Z"], dtype=np.float64)
    ny, nx = int(basin_grid["ny"]), int(basin_grid["nx"])
    xmin, xmax = float(basin_grid["xmin"]), float(basin_grid["xmax"])
    ymin, ymax = float(basin_grid["ymin"]), float(basin_grid["ymax"])
    xs = coords[:, 0]
    ys = coords[:, 1]
    if nx <= 1:
        ix = np.zeros(len(coords), dtype=np.int64)
    else:
        ix = np.rint((xs - xmin) / (xmax - xmin) * (nx - 1)).astype(np.int64)
    if ny <= 1:
        iy = np.zeros(len(coords), dtype=np.int64)
    else:
        iy = np.rint((ys - ymin) / (ymax - ymin) * (ny - 1)).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    values = z[iy, ix].astype(np.float64)
    return values


def _basin_ids_from_split_frame(frame: pd.DataFrame) -> List[int]:
    if frame is None or len(frame) == 0 or "basin_id" not in frame.columns:
        return []
    vals = pd.to_numeric(frame["basin_id"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return sorted({int(round(float(v))) for v in vals})


def apply_whole_basin_split(mineral_split: dict, basin_grid: dict) -> dict:
    """Assign whole catchments for unlabeled isolation; keep leave-one mineral labels.

    Protocol (user):
    1. Leave-one camp/fault decides which deposits are train vs test (unchanged).
    2. Any catchment containing at least one TRAIN deposit -> wholly TRAIN unlabeled.
    3. Catchments containing only TEST deposits -> wholly TEST unlabeled.
    4. A catchment is never split; train occupancy has priority over test occupancy
       for unlabeled membership (so train deposits are not surrounded by blue test U).
    """
    if mineral_split is None or basin_grid is None:
        return mineral_split
    out = dict(mineral_split)
    train_basins: set = set()
    test_basins: set = set()
    for key, bucket in (("train", train_basins), ("test", test_basins)):
        frame = out.get(key)
        if frame is None or len(frame) == 0:
            continue
        frame = frame.copy()
        xy = frame[["x", "y"]].to_numpy(dtype=np.float64)
        frame["basin_id"] = sample_basin_ids_at_coords(basin_grid, xy)
        out[key] = frame
        vals = pd.to_numeric(frame["basin_id"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        for v in vals:
            if np.isfinite(v):
                bucket.add(int(round(float(v))))

    # Train priority for unlabeled: shared basins stay train-side.
    test_basins = {b for b in test_basins if b not in train_basins}
    out["train_basin_ids"] = sorted(int(b) for b in train_basins)
    out["test_basin_ids"] = sorted(int(b) for b in test_basins)
    out["basin_split_rule"] = "whole_basin_train_priority_unlabeled"
    out["basin_absorbed_test_to_train"] = 0
    out["basin_absorbed_train_to_test"] = 0
    n_tr = int(len(out["train"])) if out.get("train") is not None else 0
    n_te = int(len(out["test"])) if out.get("test") is not None else 0
    out["train_ratio"] = float(n_tr / max(n_tr + n_te, 1))
    return out


def _attach_basin_ids_from_assignment(framed: pd.DataFrame, assignment: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = framed.copy()
    if assignment is None or "basin_id" not in getattr(assignment, "columns", []):
        return out
    mapped = _map_assignment_to_frame(out, assignment[["x", "y", "basin_id"]].copy(), id_column="basin_id")
    out["basin_id"] = pd.to_numeric(mapped["basin_id"], errors="coerce")
    return out


def _map_assignment_to_frame(
    frame: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    id_column: str,
) -> pd.DataFrame:
    """Join assignment IDs onto mineral rows by exact coords or nearest neighbor."""
    assigned = assignment.copy()
    if id_column not in assigned.columns:
        raise ValueError(f"assignment must include {id_column}.")
    same_len = len(assigned) == len(frame)
    same_xy = False
    if same_len:
        same_xy = bool(
            np.allclose(
                assigned[["x", "y"]].to_numpy(dtype=np.float64),
                frame[["x", "y"]].to_numpy(dtype=np.float64),
                equal_nan=False,
            )
        )
    mapped = frame.copy()
    if same_xy:
        mapped[id_column] = assigned[id_column].to_numpy()
    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(assigned[["x", "y"]].to_numpy(dtype=np.float64))
        _, nn = tree.query(frame[["x", "y"]].to_numpy(dtype=np.float64), k=1)
        mapped[id_column] = assigned.iloc[nn][id_column].to_numpy()
    return mapped


def map_assignment_to_minerals(
    minerals: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    id_column: str,
) -> pd.DataFrame:
    """Public, deterministic assignment join used by the reviewer protocol."""
    return _map_assignment_to_frame(_ensure_frame(minerals), assignment, id_column=id_column)


def _assign_kmeans_camps(
    frame: pd.DataFrame,
    *,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    unique_count = len(np.unique(coords, axis=0)) if len(frame) else 0
    cluster_count = max(1, min(int(n_clusters), len(frame), unique_count or 1))
    if cluster_count <= 1 or len(frame) < 3:
        cluster_ids = np.zeros(len(frame), dtype=np.int64)
        centers = np.mean(coords, axis=0, keepdims=True) if len(frame) else np.zeros((0, 2))
    else:
        kmeans = KMeans(n_clusters=cluster_count, random_state=int(random_state), n_init=10)
        cluster_ids = kmeans.fit_predict(coords).astype(np.int64)
        centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    out = frame.copy()
    out["kmeans_cluster"] = cluster_ids
    out["camp_id"] = cluster_ids
    return out, cluster_ids, centers


def _camp_centers_from_ids(coords: np.ndarray, camp_ids: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    coords = np.asarray(coords, dtype=np.float64)
    camp_ids = np.asarray(camp_ids, dtype=np.int64).reshape(-1)
    unique = sorted(int(c) for c in np.unique(camp_ids))
    centers = []
    for camp_id in unique:
        mask = camp_ids == camp_id
        if np.any(mask):
            centers.append(np.mean(coords[mask, :2], axis=0))
        else:
            centers.append(np.array([np.nan, np.nan], dtype=np.float64))
    return np.asarray(centers, dtype=np.float64), unique


def split_leave_one_camp(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 4,
    held_out_camp_index: int = 0,
    random_state: int = 42,
    camp_assignment: Optional[pd.DataFrame] = None,
) -> dict:
    """Hold out one camp as external test.

    Prefer expert ``camp_assignment`` (geological ore camps). KMeans is retained only
    for explicit legacy exploration and is not accepted by the reviewer protocol.
    """
    frame = _ensure_frame(minerals)
    empty = frame.iloc[0:0].copy()
    if len(frame) == 0:
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "n_clusters": int(n_clusters),
            "cluster_ids": [],
            "camp_ids": [],
            "held_out_camp_id": int(held_out_camp_index),
            "train_camp_ids": [],
            "test_camp_ids": [],
            "cluster_summaries": [],
            "camp_source": "empty",
            "algorithm": "leave_one_camp",
        }

    if camp_assignment is not None and len(camp_assignment) > 0:
        framed = _map_assignment_to_frame(frame, camp_assignment, id_column="camp_id")
        cluster_ids = pd.to_numeric(framed["camp_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        framed["camp_id"] = cluster_ids
        framed["kmeans_cluster"] = cluster_ids
        framed = _attach_basin_ids_from_assignment(framed, camp_assignment)
        centers, unique_ordered = _camp_centers_from_ids(
            framed[["x", "y"]].to_numpy(dtype=np.float64),
            cluster_ids,
        )
        center_id_list = [int(c) for c in unique_ordered]
        camp_source = "expert_table"
    else:
        framed, cluster_ids, centers = _assign_kmeans_camps(
            frame, n_clusters=n_clusters, random_state=random_state
        )
        center_id_list = sorted(int(c) for c in np.unique(cluster_ids))
        camp_source = "kmeans_proxy"

    unique_camps = sorted(int(c) for c in np.unique(cluster_ids))
    if not unique_camps:
        unique_camps = [0]
    held_idx = int(held_out_camp_index) % len(unique_camps)
    held_camp = unique_camps[held_idx]
    train_camps = [c for c in unique_camps if c != held_camp]
    train_rows = np.where(cluster_ids != held_camp)[0].astype(np.int64)
    test_rows = np.where(cluster_ids == held_camp)[0].astype(np.int64)
    if len(test_rows) == 0 and len(framed) > 1:
        test_rows = np.asarray([len(framed) - 1], dtype=np.int64)
        train_rows = np.asarray([i for i in range(len(framed)) if i not in set(test_rows.tolist())], dtype=np.int64)
        held_camp = int(framed.loc[int(test_rows[0]), "camp_id"])
        train_camps = [c for c in unique_camps if c != held_camp]

    summaries = []
    for camp_id in unique_camps:
        summaries.append(
            {
                "cluster_id": int(camp_id),
                "camp_id": int(camp_id),
                "sample_count": int(np.sum(cluster_ids == camp_id)),
                "train_count": int(np.sum(cluster_ids[train_rows] == camp_id)) if len(train_rows) else 0,
                "test_count": int(np.sum(cluster_ids[test_rows] == camp_id)) if len(test_rows) else 0,
                "assigned_split": "test" if camp_id == held_camp else "train",
            }
        )

    train_frame = framed.iloc[train_rows].reset_index(drop=True)
    test_frame = framed.iloc[test_rows].reset_index(drop=True)
    test_basin_ids = _basin_ids_from_split_frame(test_frame)
    train_basin_ids = _basin_ids_from_split_frame(train_frame)
    # Train priority: basins occupied by train deposits are not test basins.
    test_basin_ids = [b for b in test_basin_ids if b not in set(train_basin_ids)]

    return {
        "train": train_frame,
        "test": test_frame,
        "n_clusters": int(len(unique_camps)),
        "train_ratio": float(len(train_rows) / max(len(framed), 1)),
        "cluster_ids": cluster_ids.tolist(),
        "camp_ids": cluster_ids.tolist(),
        "cluster_centers": centers.tolist() if len(centers) else [],
        "cluster_center_ids": [int(c) for c in center_id_list],
        "held_out_camp_id": int(held_camp),
        "held_out_camp_index": int(held_idx),
        "train_camp_ids": [int(c) for c in train_camps],
        "test_camp_ids": [int(held_camp)],
        "train_cluster_ids": [int(c) for c in train_camps],
        "test_cluster_ids": [int(held_camp)],
        "train_basin_ids": [int(b) for b in train_basin_ids],
        "test_basin_ids": [int(b) for b in test_basin_ids],
        "cluster_summaries": summaries,
        "camp_source": camp_source,
        "algorithm": "leave_one_camp",
    }


def list_leave_one_camp_rounds(
    minerals: pd.DataFrame,
    *,
    n_clusters: int = 4,
    random_state: int = 42,
    camp_assignment: Optional[pd.DataFrame] = None,
) -> List[dict]:
    probe = split_leave_one_camp(
        minerals,
        n_clusters=n_clusters,
        held_out_camp_index=0,
        random_state=random_state,
        camp_assignment=camp_assignment,
    )
    n = max(1, int(probe.get("n_clusters", 1)))
    return [
        split_leave_one_camp(
            minerals,
            n_clusters=n_clusters,
            held_out_camp_index=i,
            random_state=random_state,
            camp_assignment=camp_assignment,
        )
        for i in range(n)
    ]


def load_deposit_fault_assignment(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    column_map = _normalize_column_map(frame)
    x_col = next((column_map[k] for k in ("x", "coord_x", "east", "easting") if k in column_map), None)
    y_col = next((column_map[k] for k in ("y", "coord_y", "north", "northing") if k in column_map), None)
    fault_col = next(
        (column_map[k] for k in ("fault_id", "fault", "fid", "断裂", "断裂编号") if k in column_map),
        None,
    )
    if x_col is None or y_col is None or fault_col is None:
        raise KeyError("deposit_fault_assignment CSV needs x, y, fault_id columns.")
    out = pd.DataFrame(
        {
            "x": pd.to_numeric(frame[x_col], errors="coerce"),
            "y": pd.to_numeric(frame[y_col], errors="coerce"),
            "fault_id": frame[fault_col].astype(str).str.strip(),
        }
    )
    if "deposit_id" in column_map:
        out["deposit_id"] = frame[column_map["deposit_id"]]
    else:
        out["deposit_id"] = np.arange(len(out))
    basin_col = next(
        (
            column_map[k]
            for k in ("basin_id", "basin", "catchment_id", "watershed_id", "汇水盆地", "汇水")
            if k in column_map
        ),
        None,
    )
    if basin_col is not None:
        out["basin_id"] = pd.to_numeric(frame[basin_col], errors="coerce")
    return out.dropna(subset=["x", "y"]).reset_index(drop=True)


def load_fault_lines(path: str) -> pd.DataFrame:
    """Load fault polyline CSV/TXT used for nearest-fault assignment."""
    frame = pd.read_csv(path)
    return frame


def fault_line_vertices(fault_lines: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices Nx2, fault_id per vertex)."""
    lines = fault_lines.copy()
    column_map = _normalize_column_map(lines)
    fault_col = next(
        (column_map[k] for k in ("fault_id", "fault", "fid") if k in column_map),
        None,
    )
    if fault_col is None:
        raise KeyError("fault_lines must contain fault_id.")
    lower_cols = {c.lower() for c in lines.columns}
    if {"x", "y"}.issubset(lower_cols):
        fx = pd.to_numeric(lines[column_map.get("x", "x")], errors="coerce").to_numpy(dtype=np.float64)
        fy = pd.to_numeric(lines[column_map.get("y", "y")], errors="coerce").to_numpy(dtype=np.float64)
        fids = lines[fault_col].astype(str).str.strip().to_numpy()
        vertices = np.column_stack([fx, fy])
    elif {"x1", "y1", "x2", "y2"}.issubset(lower_cols):
        x1 = pd.to_numeric(lines[column_map["x1"]], errors="coerce")
        y1 = pd.to_numeric(lines[column_map["y1"]], errors="coerce")
        x2 = pd.to_numeric(lines[column_map["x2"]], errors="coerce")
        y2 = pd.to_numeric(lines[column_map["y2"]], errors="coerce")
        fids = np.repeat(lines[fault_col].astype(str).str.strip().to_numpy(), 2)
        vertices = np.column_stack(
            [
                np.concatenate([x1.to_numpy(dtype=np.float64), x2.to_numpy(dtype=np.float64)]),
                np.concatenate([y1.to_numpy(dtype=np.float64), y2.to_numpy(dtype=np.float64)]),
            ]
        )
    else:
        raise KeyError("fault_lines need x,y,fault_id or x1,y1,x2,y2,fault_id.")
    valid = np.isfinite(vertices).all(axis=1)
    return vertices[valid], np.asarray(fids[valid], dtype=object)


def assign_coords_to_nearest_fault(
    coords: np.ndarray,
    fault_lines: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each coordinate to nearest fault_id; return (fault_ids, distances)."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        return np.array([], dtype=object), np.array([], dtype=np.float64)
    vertices, fids = fault_line_vertices(fault_lines)
    if len(vertices) == 0:
        raise ValueError("fault_lines produced no valid vertices.")
    assigned = []
    distances = []
    for xy in coords[:, :2]:
        d = np.sqrt(np.sum((vertices - xy) ** 2, axis=1))
        idx = int(np.argmin(d))
        assigned.append(str(fids[idx]))
        distances.append(float(d[idx]))
    return np.asarray(assigned, dtype=object), np.asarray(distances, dtype=np.float64)


def auto_assign_faults_from_lines(
    minerals: pd.DataFrame,
    fault_lines: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each deposit to nearest fault_id using polyline vertices."""
    frame = _ensure_frame(minerals)
    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    assigned, distances = assign_coords_to_nearest_fault(coords, fault_lines)
    out = frame.copy()
    out["fault_id"] = assigned
    out["fault_distance_m"] = distances
    out["deposit_id"] = np.arange(len(out))
    return out


def split_leave_one_fault(
    minerals: pd.DataFrame,
    *,
    held_out_fault_id: str = "",
    assignment: Optional[pd.DataFrame] = None,
    assignment_source: str = "",
    fault_lines: Optional[pd.DataFrame] = None,
) -> dict:
    frame = _ensure_frame(minerals)
    empty = frame.iloc[0:0].copy()
    source = str(assignment_source or "").strip()
    if assignment is None:
        if "fault_id" not in frame.columns:
            raise ValueError("leave_one_fault requires fault_id column or assignment table.")
        assigned = frame.copy()
        source = source or "column"
    else:
        assigned = _map_assignment_to_frame(frame, assignment, id_column="fault_id")
        assigned["fault_id"] = assigned["fault_id"].astype(str).str.strip()
        if assignment is not None and "basin_id" in assignment.columns:
            assigned = _attach_basin_ids_from_assignment(assigned, assignment)
        source = source or "table"

    if len(assigned) == 0:
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "fault_ids": [],
            "held_out_fault_id": "",
            "cluster_summaries": [],
            "assignment_source": source or "empty",
            "algorithm": "leave_one_fault",
        }

    fault_ids = sorted({str(v).strip() for v in assigned["fault_id"].tolist() if str(v).strip()})
    if not fault_ids:
        raise ValueError("No valid fault_id values found for leave_one_fault.")
    held = str(held_out_fault_id).strip() if held_out_fault_id else fault_ids[0]
    if held not in fault_ids:
        held = fault_ids[0]
    test_mask = assigned["fault_id"].astype(str).str.strip() == held
    train_rows = np.where(~test_mask.to_numpy())[0].astype(np.int64)
    test_rows = np.where(test_mask.to_numpy())[0].astype(np.int64)
    if len(test_rows) == 0:
        raise ValueError(f"Held-out fault {held} has no deposits.")
    if len(train_rows) == 0:
        raise ValueError(f"Holding out fault {held} leaves empty training set.")

    summaries = []
    for fid in fault_ids:
        mask = assigned["fault_id"].astype(str).str.strip() == fid
        summaries.append(
            {
                "fault_id": fid,
                "sample_count": int(mask.sum()),
                "train_count": int((mask & ~test_mask).sum()),
                "test_count": int((mask & test_mask).sum()),
                "assigned_split": "test" if fid == held else "train",
            }
        )

    train_frame = assigned.iloc[train_rows].reset_index(drop=True)
    test_frame = assigned.iloc[test_rows].reset_index(drop=True)
    test_basin_ids = _basin_ids_from_split_frame(test_frame)
    train_basin_ids = _basin_ids_from_split_frame(train_frame)
    test_basin_ids = [b for b in test_basin_ids if b not in set(train_basin_ids)]

    result = {
        "train": train_frame,
        "test": test_frame,
        "n_clusters": int(len(fault_ids)),
        "train_ratio": float(len(train_rows) / max(len(assigned), 1)),
        "fault_ids": fault_ids,
        "held_out_fault_id": held,
        "train_fault_ids": [f for f in fault_ids if f != held],
        "test_fault_ids": [held],
        "train_basin_ids": [int(b) for b in train_basin_ids],
        "test_basin_ids": [int(b) for b in test_basin_ids],
        "cluster_ids": assigned["fault_id"].astype(str).tolist(),
        "cluster_summaries": summaries,
        "assignment_source": source,
        "algorithm": "leave_one_fault",
    }
    if fault_lines is not None and len(fault_lines) > 0:
        try:
            vertices, vertex_fids = fault_line_vertices(fault_lines)
            result["fault_line_vertices"] = vertices.tolist()
            result["fault_line_vertex_ids"] = [str(v) for v in vertex_fids.tolist()]
        except Exception:
            pass
    return result


def list_leave_one_fault_rounds(
    minerals: pd.DataFrame,
    *,
    assignment: Optional[pd.DataFrame] = None,
    assignment_source: str = "",
    fault_lines: Optional[pd.DataFrame] = None,
) -> List[dict]:
    probe = split_leave_one_fault(
        minerals,
        held_out_fault_id="",
        assignment=assignment,
        assignment_source=assignment_source,
        fault_lines=fault_lines,
    )
    return [
        split_leave_one_fault(
            minerals,
            held_out_fault_id=fid,
            assignment=assignment,
            assignment_source=assignment_source,
            fault_lines=fault_lines,
        )
        for fid in probe.get("fault_ids", [])
    ]


def assign_coords_to_variogram_blocks(
    coords: np.ndarray,
    *,
    grid_x0: float,
    grid_y0: float,
    block_size_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ix, iy) integer grid indices for coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    block_size = float(max(block_size_m, 1.0))
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty
    ix = np.floor((coords[:, 0] - float(grid_x0)) / block_size).astype(np.int64)
    iy = np.floor((coords[:, 1] - float(grid_y0)) / block_size).astype(np.int64)
    return ix, iy


def split_variogram_block_cv(
    minerals: pd.DataFrame,
    *,
    variogram_range_m: float = 2000.0,
    held_out_block_index: int = 0,
    random_state: int = 42,
) -> dict:
    """Spatial blocks sized by dominant variogram range; hold out one block as test."""
    frame = _ensure_frame(minerals)
    empty = frame.iloc[0:0].copy()
    block_size = float(max(variogram_range_m, 1.0))
    if len(frame) == 0:
        return {
            "train": empty.copy(),
            "test": empty.copy(),
            "block_size_m": block_size,
            "held_out_block_id": 0,
            "cluster_summaries": [],
            "grid_x0": 0.0,
            "grid_y0": 0.0,
            "held_out_block_pair": [0, 0],
            "train_block_pairs": [],
            "test_block_pairs": [[0, 0]],
            "algorithm": "variogram_block_cv",
        }

    coords = frame[["x", "y"]].to_numpy(dtype=np.float64)
    x0 = float(np.min(coords[:, 0]))
    y0 = float(np.min(coords[:, 1]))
    ix, iy = assign_coords_to_variogram_blocks(
        coords, grid_x0=x0, grid_y0=y0, block_size_m=block_size
    )
    pairs = list(zip(ix.tolist(), iy.tolist()))
    unique_pairs = sorted(set(pairs))
    pair_to_id = {p: i for i, p in enumerate(unique_pairs)}
    block_ids = np.asarray([pair_to_id[p] for p in pairs], dtype=np.int64)
    framed = frame.copy()
    framed["block_id"] = block_ids
    framed["kmeans_cluster"] = block_ids
    framed["block_ix"] = ix
    framed["block_iy"] = iy
    unique_blocks = sorted(int(b) for b in np.unique(block_ids))
    held_idx = int(held_out_block_index) % max(len(unique_blocks), 1)
    held = unique_blocks[held_idx]
    train_rows = np.where(block_ids != held)[0].astype(np.int64)
    test_rows = np.where(block_ids == held)[0].astype(np.int64)
    if len(test_rows) == 0:
        rng = np.random.default_rng(int(random_state))
        test_rows = np.asarray([int(rng.integers(0, len(framed)))], dtype=np.int64)
        train_rows = np.asarray([i for i in range(len(framed)) if i not in set(test_rows.tolist())], dtype=np.int64)
        held = int(framed.loc[int(test_rows[0]), "block_id"])

    held_pair = list(unique_pairs[held]) if 0 <= held < len(unique_pairs) else [int(ix[test_rows[0]]), int(iy[test_rows[0]])]
    train_pairs = [list(unique_pairs[b]) for b in unique_blocks if b != held]
    test_pairs = [held_pair]

    summaries = []
    for bid in unique_blocks:
        pair = unique_pairs[bid]
        summaries.append(
            {
                "block_id": int(bid),
                "block_ix": int(pair[0]),
                "block_iy": int(pair[1]),
                "sample_count": int(np.sum(block_ids == bid)),
                "train_count": int(np.sum(block_ids[train_rows] == bid)) if len(train_rows) else 0,
                "test_count": int(np.sum(block_ids[test_rows] == bid)) if len(test_rows) else 0,
                "assigned_split": "test" if bid == held else "train",
            }
        )

    return {
        "train": framed.iloc[train_rows].reset_index(drop=True),
        "test": framed.iloc[test_rows].reset_index(drop=True),
        "n_clusters": int(len(unique_blocks)),
        "train_ratio": float(len(train_rows) / max(len(framed), 1)),
        "block_size_m": float(block_size),
        "variogram_range_m": float(block_size),
        "cluster_ids": block_ids.tolist(),
        "held_out_block_id": int(held),
        "held_out_block_index": int(held_idx),
        "train_block_ids": [b for b in unique_blocks if b != held],
        "test_block_ids": [int(held)],
        "train_cluster_ids": [b for b in unique_blocks if b != held],
        "test_cluster_ids": [int(held)],
        "grid_x0": float(x0),
        "grid_y0": float(y0),
        "held_out_block_pair": [int(held_pair[0]), int(held_pair[1])],
        "train_block_pairs": train_pairs,
        "test_block_pairs": test_pairs,
        "block_pair_to_id": {f"{p[0]},{p[1]}": int(i) for p, i in pair_to_id.items()},
        "cluster_summaries": summaries,
        "single_block_warning": bool(len(unique_blocks) <= 1),
        "algorithm": "variogram_block_cv",
    }


def list_variogram_block_rounds(
    minerals: pd.DataFrame,
    *,
    variogram_range_m: float = 2000.0,
    random_state: int = 42,
) -> List[dict]:
    probe = split_variogram_block_cv(
        minerals,
        variogram_range_m=variogram_range_m,
        held_out_block_index=0,
        random_state=random_state,
    )
    n = max(1, int(probe.get("n_clusters", 1)))
    return [
        split_variogram_block_cv(
            minerals,
            variogram_range_m=variogram_range_m,
            held_out_block_index=i,
            random_state=random_state,
        )
        for i in range(n)
    ]
