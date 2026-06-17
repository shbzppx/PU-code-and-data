"""Artifact packaging and manifest-driven prediction rebuild helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_utils import DETECTION_THRESHOLD
from .mineral_evaluator import MineralEvaluator


DEFAULT_THRESHOLDS = (0.30, 0.50, 0.70)


@dataclass
class ModelArtifact:
    model_path: str
    model_format: str
    manifest_path: str
    training_summary_path: str
    feature_schema_path: str


@dataclass
class PredictionArtifact:
    probability_map_path: str
    prediction_header_path: str
    zone_exports: List[Dict[str, object]]
    preview_image_path: str
    zone_statistics_path: str


class ArtifactManager:
    """Persist trained models, manifests, and prediction-zone exports."""

    def __init__(
        self,
        wrapper_factory: Callable[..., object],
        *,
        thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    ) -> None:
        self.wrapper_factory = wrapper_factory
        self.thresholds = tuple(float(value) for value in thresholds)

    def save_run_exports(
        self,
        output_dir: str,
        ranked_results: Sequence[Dict[str, object]],
        *,
        report_path: Optional[str] = None,
        stage_results: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[str, str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        leaderboard_df = pd.DataFrame([self._leaderboard_row(item) for item in ranked_results])
        leaderboard_path = root / "leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")

        trials_rows: List[Dict[str, object]] = []
        for result in ranked_results:
            for history in result.get("optimization_history") or []:
                row = {
                    "model_name": result.get("model_name"),
                    "stage": result.get("stage", "stage2"),
                    "scheme_cache_key": result.get("dataset_params", {}).get("cache_key", ""),
                }
                row.update(history)
                trials_rows.append(row)
        for result in stage_results or []:
            for history in result.get("optimization_history") or []:
                row = {
                    "model_name": result.get("model_name"),
                    "stage": result.get("stage", "stage1"),
                    "scheme_cache_key": result.get("dataset_params", {}).get("cache_key", ""),
                }
                row.update(history)
                trials_rows.append(row)
        all_trials_path = root / "all_trials.csv"
        pd.DataFrame(trials_rows).to_csv(all_trials_path, index=False, encoding="utf-8-sig")

        manifest_index_path = root / "model_packages.json"
        manifests = [
            {
                "rank": index + 1,
                "model_name": item.get("model_name"),
                "composite_score": item.get("composite_score"),
                "model_artifact_path": item.get("model_artifact_path"),
                "rebuild_manifest_path": item.get("rebuild_manifest_path"),
            }
            for index, item in enumerate(ranked_results)
        ]
        self._write_json(manifest_index_path, manifests)

        output = {
            "leaderboard_path": str(leaderboard_path),
            "all_trials_path": str(all_trials_path),
            "manifest_index_path": str(manifest_index_path),
        }
        if report_path:
            output["report_path"] = report_path
        return output

    def package_trained_result(
        self,
        result: Dict[str, object],
        wrapper: object,
        evaluator: MineralEvaluator,
        *,
        output_dir: str,
        val_minerals_df=None,
        test_minerals_df=None,
        runtime_mode: str = "practical",
        software_version: str = "model_comparison_v2",
    ) -> Dict[str, object]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_source = "evaluated_checkpoint"

        model_extension = ".pth" if self._is_deep_learning_result(result) else ".joblib"
        model_path = root / f"best_model{model_extension}"
        wrapper.save_model(str(model_path))

        probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
        probability_map_path = root / "probability_map.npy"
        np.save(probability_map_path, probability_map)
        best_detection_threshold = float(result.get("best_detection_threshold", DETECTION_THRESHOLD) or DETECTION_THRESHOLD)

        prediction_header = self._build_prediction_header(evaluator, result)
        prediction_header_path = root / "prediction_header.json"
        self._write_json(prediction_header_path, prediction_header)

        preview_image_path = root / "probability_map.png"
        self._save_probability_preview(preview_image_path, probability_map)

        zones_dir = root / "zones"
        zones_dir.mkdir(parents=True, exist_ok=True)
        export_thresholds = sorted({float(value) for value in list(self.thresholds) + [best_detection_threshold]})
        zone_exports = self._export_threshold_zones(
            zones_dir,
            probability_map,
            evaluator,
            val_minerals_df=val_minerals_df,
            test_minerals_df=test_minerals_df,
            thresholds=export_thresholds,
        )
        zone_statistics_path = root / "zone_statistics.csv"
        pd.DataFrame(zone_exports).to_csv(zone_statistics_path, index=False, encoding="utf-8-sig")

        dataset_params = dict(result.get("dataset_params") or {})
        best_params = dict(result.get("best_params") or {})
        workflow_config = dict(result.get("workflow_config") or {})
        normalization_stats = getattr(wrapper, "normalization_stats", None) or result.get("dataset_summary", {}).get("normalization_stats")
        split_info = getattr(wrapper, "split_info", None) or result.get("dataset_summary", {}).get("split_info")
        best_params_path = root / "best_params.json"
        dataset_scheme_path = root / "dataset_scheme.json"
        training_summary_path = root / "training_summary.json"
        feature_schema_path = root / "feature_schema.json"
        self._write_json(best_params_path, best_params)
        self._write_json(dataset_scheme_path, dataset_params)
        self._write_json(
            training_summary_path,
            {
                "model_name": result.get("model_name"),
                "model_type": result.get("model_type"),
                "training_time": result.get("training_time"),
                "val_accuracy": result.get("val_accuracy"),
                "val_f1": result.get("val_f1"),
                "val_mineral_detection_rate": result.get("val_mineral_detection_rate"),
                "test_accuracy": result.get("test_accuracy"),
                "test_f1": result.get("test_f1"),
                "test_mineral_detection_rate": result.get("test_mineral_detection_rate"),
                "composite_score": result.get("composite_score"),
                "stage": result.get("stage", "stage2"),
                "checkpoint_source": checkpoint_source,
                "workflow_config": workflow_config,
                "best_detection_threshold": best_detection_threshold,
                "normalization_stats": normalization_stats,
                "split_info": split_info,
            },
        )
        self._write_json(
            feature_schema_path,
            {
                "dataset_meta": result.get("dataset_meta"),
                "dataset_summary": result.get("dataset_summary"),
                "label_mapping": result.get("dataset_summary", {}).get("label_mapping"),
                "normalization_stats": normalization_stats,
                "split_info": split_info,
            },
        )

        manifest_path = root / "rebuild_manifest.json"
        manifest = {
            "model_name": result.get("model_name"),
            "model_type": result.get("model_type"),
            "model_path": str(model_path),
            "model_format": model_extension.lstrip("."),
            "h5_path": result.get("h5_path"),
            "h5_mode": result.get("dataset_summary", {}).get("h5_mode"),
            "input_channels": result.get("dataset_meta", {}).get("input_channels"),
            "image_size": result.get("dataset_meta", {}).get("image_size"),
            "num_classes": result.get("dataset_meta", {}).get("num_classes"),
            "dataset_params": dataset_params,
            "best_params": best_params,
            "workflow_config": workflow_config,
            "label_mapping": result.get("dataset_summary", {}).get("label_mapping"),
            "class_mapping": result.get("dataset_summary", {}).get("class_distribution"),
            "normalization_stats": normalization_stats,
            "split_info": split_info,
            "prediction_mode": "probability_map",
            "checkpoint_source": checkpoint_source,
            "thresholds": export_thresholds,
            "default_detection_threshold": best_detection_threshold,
            "probability_map_path": str(probability_map_path),
            "prediction_header_path": str(prediction_header_path),
            "zone_statistics_path": str(zone_statistics_path),
            "feature_schema_path": str(feature_schema_path),
            "training_summary_path": str(training_summary_path),
            "software_version": software_version,
            "runtime_mode": runtime_mode,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_json(manifest_path, manifest)

        model_artifact = ModelArtifact(
            model_path=str(model_path),
            model_format=model_extension.lstrip("."),
            manifest_path=str(manifest_path),
            training_summary_path=str(training_summary_path),
            feature_schema_path=str(feature_schema_path),
        )
        prediction_artifact = PredictionArtifact(
            probability_map_path=str(probability_map_path),
            prediction_header_path=str(prediction_header_path),
            zone_exports=zone_exports,
            preview_image_path=str(preview_image_path),
            zone_statistics_path=str(zone_statistics_path),
        )
        return {
            "model_artifact": asdict(model_artifact),
            "prediction_artifact": asdict(prediction_artifact),
            "model_artifact_path": str(model_path),
            "rebuild_manifest_path": str(manifest_path),
            "probability_map_path": str(probability_map_path),
            "zone_statistics_path": str(zone_statistics_path),
        }

    def rebuild_from_manifest(self, manifest_path: str, output_dir: str) -> Dict[str, object]:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        model_name = str(manifest["model_name"])
        dataset_meta = {
            "input_channels": int(manifest.get("input_channels") or 1),
            "image_size": int(manifest.get("image_size") or 1),
            "num_classes": int(manifest.get("num_classes") or 2),
        }
        wrapper = self.wrapper_factory(model_name, dataset_meta, profile="rebuild")
        wrapper.load_model(str(manifest["model_path"]))

        evaluator = MineralEvaluator(
            str(manifest["h5_path"]),
            buffer_radius=float(manifest.get("dataset_params", {}).get("buffer_radius", 500)),
            prediction_patch_size=int(manifest.get("dataset_params", {}).get("patch_size", dataset_meta["image_size"])),
            detection_threshold=float(manifest.get("default_detection_threshold", DETECTION_THRESHOLD)),
        )

        rebuild_root = Path(output_dir)
        rebuild_root.mkdir(parents=True, exist_ok=True)
        probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
        probability_map_path = rebuild_root / "probability_map.npy"
        np.save(probability_map_path, probability_map)

        prediction_header = self._build_prediction_header(
            evaluator,
            {
                "dataset_params": manifest.get("dataset_params", {}),
                "best_detection_threshold": manifest.get("default_detection_threshold", DETECTION_THRESHOLD),
            },
        )
        prediction_header_path = rebuild_root / "prediction_header.json"
        self._write_json(prediction_header_path, prediction_header)
        preview_image_path = rebuild_root / "probability_map.png"
        self._save_probability_preview(preview_image_path, probability_map)

        zones_dir = rebuild_root / "zones"
        zones_dir.mkdir(parents=True, exist_ok=True)
        zone_exports = self._export_threshold_zones(
            zones_dir,
            probability_map,
            evaluator,
            val_minerals_df=None,
            test_minerals_df=None,
            thresholds=manifest.get("thresholds") or self.thresholds,
        )
        zone_statistics_path = rebuild_root / "zone_statistics.csv"
        pd.DataFrame(zone_exports).to_csv(zone_statistics_path, index=False, encoding="utf-8-sig")
        return {
            "manifest_path": str(manifest_path),
            "output_dir": str(rebuild_root),
            "probability_map_path": str(probability_map_path),
            "prediction_header_path": str(prediction_header_path),
            "preview_image_path": str(preview_image_path),
            "zone_statistics_path": str(zone_statistics_path),
            "zone_exports": zone_exports,
        }

    def _export_threshold_zones(
        self,
        zones_dir: Path,
        probability_map: np.ndarray,
        evaluator: MineralEvaluator,
        *,
        val_minerals_df,
        test_minerals_df,
        thresholds: Optional[Iterable[float]] = None,
    ) -> List[Dict[str, object]]:
        zone_exports: List[Dict[str, object]] = []
        for threshold in thresholds or self.thresholds:
            mask = probability_map >= float(threshold)
            stem = zones_dir / f"threshold_{float(threshold):.2f}"
            np.save(stem.with_suffix(".npy"), mask.astype(np.uint8))
            self._save_zone_preview(stem.with_suffix(".png"), mask)
            val_detection = evaluator.evaluate_probability_map(probability_map, val_minerals_df, threshold=float(threshold))
            test_detection = evaluator.evaluate_probability_map(probability_map, test_minerals_df, threshold=float(threshold))
            stats = self._compute_zone_stats(mask, evaluator.metadata, threshold, val_detection, test_detection)
            stats_path = stem.with_name(stem.name + "_stats.json")
            self._write_json(stats_path, stats)
            stats["mask_path"] = str(stem.with_suffix(".npy"))
            stats["preview_path"] = str(stem.with_suffix(".png"))
            stats["stats_path"] = str(stats_path)
            zone_exports.append(stats)
        return zone_exports

    def _build_prediction_header(self, evaluator: MineralEvaluator, result: Dict[str, object]) -> Dict[str, object]:
        metadata = {key: self._to_jsonable(value) for key, value in evaluator.metadata.items()}
        return {
            "metadata": metadata,
            "dataset_params": self._to_jsonable(result.get("dataset_params", {})),
            "best_detection_threshold": self._to_jsonable(result.get("best_detection_threshold", DETECTION_THRESHOLD)),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _compute_zone_stats(
        self,
        mask: np.ndarray,
        metadata: Dict[str, object],
        threshold: float,
        val_detection: Optional[Dict[str, object]],
        test_detection: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        positive_pixels = int(mask.sum())
        total_pixels = int(mask.size)
        area_ratio = float(positive_pixels / total_pixels) if total_pixels else 0.0

        nx = max(int(metadata.get("nx", mask.shape[1])), 1)
        ny = max(int(metadata.get("ny", mask.shape[0])), 1)
        x_min = float(metadata.get("x_min", 0.0))
        x_max = float(metadata.get("x_max", float(nx - 1)))
        y_min = float(metadata.get("y_min", 0.0))
        y_max = float(metadata.get("y_max", float(ny - 1)))
        dx = abs(x_max - x_min) / max(nx - 1, 1)
        dy = abs(y_max - y_min) / max(ny - 1, 1)
        pixel_area = dx * dy

        return {
            "threshold": float(threshold),
            "zone_count": int(self._count_connected_components(mask)),
            "pixel_count": positive_pixels,
            "total_pixel_count": total_pixels,
            "area_ratio": area_ratio,
            "area_estimate": float(positive_pixels * pixel_area),
            "val_mineral_count": int((val_detection or {}).get("mineral_count", 0)),
            "val_detected_count": int((val_detection or {}).get("detected_count", 0)),
            "val_detection_rate": float((val_detection or {}).get("detection_rate", 0.0)),
            "test_mineral_count": int((test_detection or {}).get("mineral_count", 0)),
            "test_detected_count": int((test_detection or {}).get("detected_count", 0)),
            "test_detection_rate": float((test_detection or {}).get("detection_rate", 0.0)),
        }

    def _leaderboard_row(self, result: Dict[str, object]) -> Dict[str, object]:
        dataset_params = dict(result.get("dataset_params") or {})
        return {
            "model_name": result.get("model_name", ""),
            "composite_score": result.get("composite_score"),
            "val_accuracy": result.get("val_accuracy"),
            "val_f1": result.get("val_f1"),
            "val_mineral_detection_rate": result.get("val_mineral_detection_rate"),
            "test_accuracy": result.get("test_accuracy"),
            "test_f1": result.get("test_f1"),
            "test_mineral_detection_rate": result.get("test_mineral_detection_rate"),
            "best_detection_threshold": result.get("best_detection_threshold"),
            "training_time": result.get("training_time"),
            "patch_size": dataset_params.get("patch_size"),
            "patch_stride": dataset_params.get("patch_stride"),
            "buffer_radius": dataset_params.get("buffer_radius"),
            "cache_key": dataset_params.get("cache_key", ""),
            "model_artifact_path": result.get("model_artifact_path", ""),
            "rebuild_manifest_path": result.get("rebuild_manifest_path", ""),
        }

    def _save_probability_preview(self, output_path: Path, probability_map: np.ndarray) -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(probability_map, cmap="viridis")
        plt.colorbar(label="Probability")
        plt.title("Probability Map")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _save_zone_preview(self, output_path: Path, zone_mask: np.ndarray) -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(zone_mask.astype(np.uint8), cmap="magma")
        plt.title("Prediction Zone")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    def _count_connected_components(self, mask: np.ndarray) -> int:
        if mask.size == 0:
            return 0

        visited = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        component_count = 0
        for row in range(height):
            for col in range(width):
                if visited[row, col] or not mask[row, col]:
                    continue
                component_count += 1
                self._flood_fill(mask, visited, row, col)
        return component_count

    def _flood_fill(self, mask: np.ndarray, visited: np.ndarray, start_row: int, start_col: int) -> None:
        stack = [(start_row, start_col)]
        height, width = mask.shape
        while stack:
            row, col = stack.pop()
            if row < 0 or row >= height or col < 0 or col >= width:
                continue
            if visited[row, col] or not mask[row, col]:
                continue
            visited[row, col] = True
            stack.extend(
                [
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ]
            )

    def _write_json(self, output_path: Path, payload: object) -> None:
        output_path.write_text(
            json.dumps(self._to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _to_jsonable(self, value: object):
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _is_deep_learning_result(self, result: Dict[str, object]) -> bool:
        model_name = str(result.get("model_name") or "")
        return model_name.startswith("CNN") or model_name.startswith("ResNet")
