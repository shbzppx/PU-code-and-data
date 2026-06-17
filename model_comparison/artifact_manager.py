"""Artifact packaging and manifest-driven prediction rebuild helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_utils import DETECTION_THRESHOLD, primary_test_ei_score
from .metric_protocol import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_THRESHOLD_STEP,
    METRIC_PROTOCOL,
    PAF_SCOPE,
    THRESHOLD_RULE,
    THRESHOLD_STRATEGY,
    detection_from_metric_row,
    evaluate_independent_test_metrics,
    metric_protocol_fields,
)
from .mineral_evaluator import MineralEvaluator
from .model_wrappers import SYSTEM_MODEL_SPECS


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
            trial_context = self._build_split_summary_fields(result)
            trial_context.update(self._build_evaluation_summary_fields(result))
            trial_context.update(self._build_model_context_fields(result))
            for history in result.get("optimization_history") or []:
                row = {
                    "model_name": result.get("model_name"),
                    "stage": result.get("stage", "stage2"),
                    "scheme_cache_key": result.get("dataset_params", {}).get("cache_key", ""),
                }
                row.update(
                    {
                        "split_mode": trial_context.get("split_mode", ""),
                        "spatial_cluster_active": trial_context.get("spatial_cluster_active", False),
                        "spatial_cv_enabled": trial_context.get("spatial_cv_enabled", False),
                        "independent_test_sample_count": trial_context.get("independent_test_sample_count", 0),
                        "independent_test_mineral_count": trial_context.get("independent_test_mineral_count", 0),
                        "independent_test_accuracy": trial_context.get("independent_test_accuracy"),
                        "independent_test_f1": trial_context.get("independent_test_f1"),
                        "independent_test_mineral_detection_rate": trial_context.get("independent_test_mineral_detection_rate"),
                        "model_family": trial_context.get("model_family", ""),
                        "input_kind": trial_context.get("input_kind", ""),
                        "label_mode": trial_context.get("label_mode", ""),
                        "training_mode": trial_context.get("training_mode", ""),
                        "loss_type": trial_context.get("loss_type", ""),
                        "prior_mode": trial_context.get("prior_mode", ""),
                        "prior": trial_context.get("prior", ""),
                        "beta": trial_context.get("beta", ""),
                        "gamma": trial_context.get("gamma", ""),
                        "adaptive_window": trial_context.get("adaptive_window", ""),
                        "adaptive_lambda": trial_context.get("adaptive_lambda", ""),
                        "adaptive_gamma_min": trial_context.get("adaptive_gamma_min", ""),
                        "adaptive_gamma_max": trial_context.get("adaptive_gamma_max", ""),
                        "learning_rate": trial_context.get("learning_rate", ""),
                        "batch_size": trial_context.get("batch_size", ""),
                        "optimizer": trial_context.get("optimizer", ""),
                    }
                )
                row.update(history)
                trials_rows.append(row)
        for result in stage_results or []:
            trial_context = self._build_split_summary_fields(result)
            trial_context.update(self._build_evaluation_summary_fields(result))
            trial_context.update(self._build_model_context_fields(result))
            for history in result.get("optimization_history") or []:
                row = {
                    "model_name": result.get("model_name"),
                    "stage": result.get("stage", "stage1"),
                    "scheme_cache_key": result.get("dataset_params", {}).get("cache_key", ""),
                }
                row.update(
                    {
                        "split_mode": trial_context.get("split_mode", ""),
                        "spatial_cluster_active": trial_context.get("spatial_cluster_active", False),
                        "spatial_cv_enabled": trial_context.get("spatial_cv_enabled", False),
                        "independent_test_sample_count": trial_context.get("independent_test_sample_count", 0),
                        "independent_test_mineral_count": trial_context.get("independent_test_mineral_count", 0),
                        "independent_test_accuracy": trial_context.get("independent_test_accuracy"),
                        "independent_test_f1": trial_context.get("independent_test_f1"),
                        "independent_test_mineral_detection_rate": trial_context.get("independent_test_mineral_detection_rate"),
                        "model_family": trial_context.get("model_family", ""),
                        "input_kind": trial_context.get("input_kind", ""),
                        "label_mode": trial_context.get("label_mode", ""),
                        "training_mode": trial_context.get("training_mode", ""),
                        "loss_type": trial_context.get("loss_type", ""),
                        "prior_mode": trial_context.get("prior_mode", ""),
                        "prior": trial_context.get("prior", ""),
                        "beta": trial_context.get("beta", ""),
                        "gamma": trial_context.get("gamma", ""),
                        "adaptive_window": trial_context.get("adaptive_window", ""),
                        "adaptive_lambda": trial_context.get("adaptive_lambda", ""),
                        "adaptive_gamma_min": trial_context.get("adaptive_gamma_min", ""),
                        "adaptive_gamma_max": trial_context.get("adaptive_gamma_max", ""),
                        "learning_rate": trial_context.get("learning_rate", ""),
                        "batch_size": trial_context.get("batch_size", ""),
                        "optimizer": trial_context.get("optimizer", ""),
                    }
                )
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
                "primary_score_test_ei": primary_test_ei_score(item),
                "metric_protocol": item.get("metric_protocol", METRIC_PROTOCOL),
                "threshold_strategy": item.get("threshold_strategy", THRESHOLD_STRATEGY),
                "paf_scope": item.get("paf_scope", PAF_SCOPE),
                "threshold_rule": item.get("threshold_rule", THRESHOLD_RULE),
                "model_artifact_path": item.get("model_artifact_path"),
                "training_compatible_model_path": item.get("training_compatible_model_path"),
                "rebuild_manifest_path": item.get("rebuild_manifest_path"),
                "params_path": item.get("params_path"),
                "normalization_params_path": item.get("normalization_params_path"),
                "test_area_path": item.get("test_area_path"),
                "test_mineral_path": item.get("test_mineral_path"),
                **self._build_model_context_fields(item),
                "split_mode": self._build_split_summary_fields(item).get("split_mode", ""),
                "spatial_cluster_active": self._build_split_summary_fields(item).get("spatial_cluster_active", False),
                "spatial_cluster_n_clusters": self._build_split_summary_fields(item).get("spatial_cluster_n_clusters", 0),
                "spatial_cv_enabled": self._build_split_summary_fields(item).get("spatial_cv_enabled", False),
                "spatial_cv_fold_count": self._build_split_summary_fields(item).get("spatial_cv_fold_count", 0),
                "independent_test_sample_count": self._build_evaluation_summary_fields(item).get("independent_test_sample_count", 0),
                "independent_test_mineral_count": self._build_evaluation_summary_fields(item).get("independent_test_mineral_count", 0),
                "independent_test_accuracy": self._build_evaluation_summary_fields(item).get("independent_test_accuracy"),
                "independent_test_f1": self._build_evaluation_summary_fields(item).get("independent_test_f1"),
                "independent_test_mineral_detection_rate": self._build_evaluation_summary_fields(item).get("independent_test_mineral_detection_rate"),
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

    def _build_split_summary_fields(self, result: Dict[str, object]) -> Dict[str, object]:
        dataset_params = dict(result.get("dataset_params") or {})
        dataset_summary = dict(result.get("dataset_summary") or {})
        split_info = dict(dataset_summary.get("split_info") or dataset_params.get("split_info") or {})

        def _pick(*values, default=None):
            for value in values:
                if value not in (None, ""):
                    return value
            return default

        def _as_int(*values, default=0):
            value = _pick(*values, default=default)
            try:
                return int(value)
            except (TypeError, ValueError):
                return int(default)

        def _as_float(*values, default=0.0):
            value = _pick(*values, default=default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        split_mode = str(
            _pick(
                dataset_params.get("split_mode"),
                dataset_summary.get("split_mode"),
                split_info.get("effective_strategy"),
                split_info.get("strategy"),
                "",
                default="",
            )
        ).strip().lower()

        split_fields = {
            "split_mode": split_mode,
            "train_sample_count": _as_int(dataset_summary.get("train_sample_count"), default=0),
            "val_sample_count": _as_int(dataset_summary.get("val_sample_count"), default=0),
            "test_sample_count": _as_int(dataset_summary.get("test_sample_count"), default=0),
            "train_mineral_count": _as_int(dataset_summary.get("train_mineral_count"), default=0),
            "val_mineral_count": _as_int(dataset_summary.get("val_mineral_count"), default=0),
            "test_mineral_count": _as_int(dataset_summary.get("test_mineral_count"), default=0),
            "spatial_region_active": bool(_pick(dataset_params.get("spatial_region_active"), dataset_summary.get("spatial_region_active"), default=False)),
            "spatial_region_train_bounds": dict(_pick(dataset_params.get("spatial_region_train_bounds"), dataset_summary.get("spatial_region_train_bounds"), default={}) or {}),
            "spatial_region_test_bounds": dict(_pick(dataset_params.get("spatial_region_test_bounds"), dataset_summary.get("spatial_region_test_bounds"), default={}) or {}),
            "spatial_region_buffer_distance": _as_float(dataset_params.get("spatial_region_buffer_distance"), dataset_summary.get("spatial_region_buffer_distance"), default=0.0),
            "spatial_region_train_sample_count": _as_int(dataset_params.get("spatial_region_train_sample_count"), dataset_summary.get("spatial_region_train_sample_count"), default=0),
            "spatial_region_test_sample_count": _as_int(dataset_params.get("spatial_region_test_sample_count"), dataset_summary.get("spatial_region_test_sample_count"), default=0),
            "spatial_region_gray_sample_count": _as_int(dataset_params.get("spatial_region_gray_sample_count"), dataset_summary.get("spatial_region_gray_sample_count"), default=0),
            "spatial_region_outside_sample_count": _as_int(dataset_params.get("spatial_region_outside_sample_count"), dataset_summary.get("spatial_region_outside_sample_count"), default=0),
            "spatial_region_overlap_sample_count": _as_int(dataset_params.get("spatial_region_overlap_sample_count"), dataset_summary.get("spatial_region_overlap_sample_count"), default=0),
            "spatial_region_train_no_ore_sample_count": _as_int(dataset_params.get("spatial_region_train_no_ore_sample_count"), dataset_summary.get("spatial_region_train_no_ore_sample_count"), default=0),
            "spatial_region_test_no_ore_sample_count": _as_int(dataset_params.get("spatial_region_test_no_ore_sample_count"), dataset_summary.get("spatial_region_test_no_ore_sample_count"), default=0),
            "spatial_cluster_active": bool(_pick(dataset_params.get("spatial_cluster_active"), dataset_summary.get("spatial_cluster_active"), default=False)),
            "spatial_cluster_n_clusters": _as_int(dataset_params.get("spatial_cluster_n_clusters"), dataset_summary.get("spatial_cluster_n_clusters"), default=0),
            "spatial_cluster_train_ratio": _as_float(dataset_params.get("spatial_cluster_train_ratio"), dataset_summary.get("spatial_cluster_train_ratio"), default=0.0),
            "spatial_cluster_cv_folds": _as_int(dataset_params.get("spatial_cluster_cv_folds"), dataset_summary.get("spatial_cluster_cv_folds"), default=0),
            "spatial_cluster_random_state": _as_int(dataset_params.get("spatial_cluster_random_state"), dataset_summary.get("spatial_cluster_random_state"), default=0),
            "spatial_cluster_train_mineral_count": _as_int(dataset_params.get("spatial_cluster_train_mineral_count"), dataset_summary.get("spatial_cluster_train_mineral_count"), default=0),
            "spatial_cluster_test_mineral_count": _as_int(dataset_params.get("spatial_cluster_test_mineral_count"), dataset_summary.get("spatial_cluster_test_mineral_count"), default=0),
            "spatial_cv_enabled": bool(_pick(dataset_params.get("spatial_cv_enabled"), dataset_summary.get("spatial_cv_enabled"), default=False)),
            "spatial_cv_fold_count": _as_int(dataset_params.get("spatial_cv_fold_count"), dataset_summary.get("spatial_cv_fold_count"), default=0),
            "spatial_cv_axis": _pick(dataset_params.get("spatial_cv_axis"), dataset_summary.get("spatial_cv_axis"), default=None),
            "spatial_cv_buffer_distance": _as_float(dataset_params.get("spatial_cv_buffer_distance"), dataset_summary.get("spatial_cv_buffer_distance"), default=0.0),
            "spatial_cv_selected_fold": _as_int(dataset_params.get("spatial_cv_selected_fold"), dataset_summary.get("spatial_cv_selected_fold"), default=0),
        }
        split_fields["independent_test_sample_count"] = split_fields["test_sample_count"]
        split_fields["independent_test_mineral_count"] = split_fields["test_mineral_count"]
        return split_fields

    def _build_model_context_fields(self, result: Dict[str, object]) -> Dict[str, object]:
        context = dict(result.get("model_context") or {})
        training_config = dict(result.get("training_config") or result.get("best_params") or result.get("config") or {})

        def _pick(*values, default=None):
            for value in values:
                if value not in (None, ""):
                    return value
            return default

        model_fields = {
            "model_family": _pick(context.get("model_family"), default=""),
            "input_kind": _pick(context.get("input_kind"), default=""),
            "label_mode": _pick(context.get("label_mode"), default=""),
            "training_mode": _pick(context.get("training_mode"), default=""),
            "is_pu_model": bool(_pick(context.get("is_pu_model"), default=False)),
            "loss_type": _pick(training_config.get("loss_type"), default=""),
            "prior_mode": _pick(training_config.get("prior_mode"), default=""),
            "prior": _pick(training_config.get("prior"), default=""),
            "beta": _pick(training_config.get("beta"), default=""),
            "gamma": _pick(training_config.get("gamma"), default=""),
            "adaptive_window": _pick(training_config.get("adaptive_window"), default=""),
            "adaptive_lambda": _pick(training_config.get("adaptive_lambda"), default=""),
            "adaptive_gamma_min": _pick(training_config.get("adaptive_gamma_min"), default=""),
            "adaptive_gamma_max": _pick(training_config.get("adaptive_gamma_max"), default=""),
            "learning_rate": _pick(training_config.get("learning_rate"), training_config.get("lr"), default=""),
            "batch_size": _pick(training_config.get("batch_size"), default=""),
            "optimizer": _pick(training_config.get("optimizer"), default=""),
        }
        return model_fields

    def _build_evaluation_summary_fields(self, result: Dict[str, object]) -> Dict[str, object]:
        split_fields = self._build_split_summary_fields(result)
        test_accuracy = result.get("test_accuracy")
        test_f1 = result.get("test_f1")
        test_detection_rate = result.get("test_mineral_detection_rate")
        return {
            "test_accuracy": test_accuracy,
            "test_f1": test_f1,
            "test_mineral_detection_rate": test_detection_rate,
            "test_sr": result.get("test_sr"),
            "test_paf": result.get("test_paf"),
            "test_ei": result.get("test_ei"),
            "val_sr": result.get("val_sr"),
            "val_paf": result.get("val_paf"),
            "val_ei": result.get("val_ei"),
            "cv_ei_mean": result.get("cv_ei_mean"),
            "cv_ei_std": result.get("cv_ei_std"),
            "independent_test_accuracy": test_accuracy,
            "independent_test_f1": test_f1,
            "independent_test_mineral_detection_rate": test_detection_rate,
            "independent_test_sample_count": split_fields.get("independent_test_sample_count", 0),
            "independent_test_mineral_count": split_fields.get("independent_test_mineral_count", 0),
        }

    def package_trained_result(
        self,
        result: Dict[str, object],
        wrapper: object,
        evaluator: MineralEvaluator,
        *,
        output_dir: str,
        val_minerals_df=None,
        test_minerals_df=None,
        test_area_positions=None,
        runtime_mode: str = "practical",
        software_version: str = "model_comparison_v2",
    ) -> Dict[str, object]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_source = "evaluated_checkpoint"

        model_extension = ".pth" if self._is_deep_learning_result(result) else ".joblib"
        model_path = root / f"best_model{model_extension}"
        wrapper.save_model(str(model_path))
        compatibility_model_path = self._save_training_model_alias(root, model_path, model_extension)

        probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
        probability_map_path = root / "probability_map.npy"
        np.save(probability_map_path, probability_map)
        best_detection_threshold = float(result.get("best_detection_threshold", DETECTION_THRESHOLD) or DETECTION_THRESHOLD)
        best_test_threshold = result.get("best_test_threshold")
        best_test_threshold = (
            float(best_test_threshold)
            if best_test_threshold not in (None, "")
            else best_detection_threshold
        )
        result = dict(result)
        result["best_detection_threshold"] = best_detection_threshold
        result["best_test_threshold"] = best_test_threshold
        protocol = self._metric_protocol_config(result)
        test_area_positions = self._resolve_test_area_positions(
            test_area_positions,
            result,
            evaluator,
        )

        prediction_header = self._build_prediction_header(evaluator, result)
        prediction_header_path = root / "prediction_header.json"
        self._write_json(prediction_header_path, prediction_header)

        preview_image_path = root / "probability_map.png"
        self._save_probability_preview(preview_image_path, probability_map, metadata=evaluator.metadata)

        zones_dir = root / "zones"
        zones_dir.mkdir(parents=True, exist_ok=True)
        export_thresholds = sorted({float(value) for value in list(self.thresholds) + [best_detection_threshold, best_test_threshold]})
        zone_exports = self._export_threshold_zones(
            zones_dir,
            probability_map,
            evaluator,
            val_minerals_df=val_minerals_df,
            test_minerals_df=test_minerals_df,
            test_area_positions=test_area_positions,
            threshold_step=protocol["threshold_step"],
            distance_threshold=protocol["distance_threshold"],
            thresholds=export_thresholds,
        )
        zone_statistics_path = root / "zone_statistics.csv"
        pd.DataFrame(zone_exports).to_csv(zone_statistics_path, index=False, encoding="utf-8-sig")

        dataset_params = dict(result.get("dataset_params") or {})
        best_params = dict(result.get("best_params") or {})
        workflow_config = dict(result.get("workflow_config") or {})
        model_context = self._build_model_context_fields(result)
        split_summary = self._build_split_summary_fields(result)
        evaluation_summary = self._build_evaluation_summary_fields(result)
        normalization_stats = getattr(wrapper, "normalization_stats", None) or result.get("dataset_summary", {}).get("normalization_stats")
        split_info = getattr(wrapper, "split_info", None) or result.get("dataset_summary", {}).get("split_info")
        training_config = dict(result.get("training_config") or result.get("best_params") or result.get("config") or {})
        best_params_path = root / "best_params.json"
        dataset_scheme_path = root / "dataset_scheme.json"
        training_summary_path = root / "training_summary.json"
        feature_schema_path = root / "feature_schema.json"
        self._write_json(best_params_path, best_params)
        self._write_json(dataset_scheme_path, dataset_params)
        self._write_json(
            training_summary_path,
            {
                "auto_optimization_rank": result.get("auto_optimization_rank"),
                "model_name": result.get("model_name"),
                "model_type": result.get("model_type"),
                "training_time": result.get("training_time"),
                "val_accuracy": result.get("val_accuracy"),
                "val_f1": result.get("val_f1"),
                "val_mineral_detection_rate": result.get("val_mineral_detection_rate"),
                "test_accuracy": result.get("test_accuracy"),
                "test_f1": result.get("test_f1"),
                "test_mineral_detection_rate": result.get("test_mineral_detection_rate"),
                "independent_test_accuracy": evaluation_summary.get("independent_test_accuracy"),
                "independent_test_f1": evaluation_summary.get("independent_test_f1"),
                "independent_test_mineral_detection_rate": evaluation_summary.get("independent_test_mineral_detection_rate"),
                "composite_score": result.get("composite_score"),
                "stage": result.get("stage", "stage2"),
                "checkpoint_source": checkpoint_source,
                "workflow_config": workflow_config,
                "best_detection_threshold": best_detection_threshold,
                "best_test_threshold": best_test_threshold,
                "best_test_sr": result.get("best_test_sr"),
                "best_test_paf": result.get("best_test_paf"),
                "best_test_ei": result.get("best_test_ei"),
                **metric_protocol_fields(
                    threshold_step=protocol["threshold_step"],
                    distance_threshold=protocol["distance_threshold"],
                ),
                "normalization_stats": normalization_stats,
                "split_info": split_info,
                "split_summary": split_summary,
                "evaluation_summary": evaluation_summary,
                "model_context": model_context,
                "training_config": training_config,
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
                "split_summary": split_summary,
                "evaluation_summary": evaluation_summary,
                "model_context": model_context,
                "training_config": training_config,
            },
        )

        compatibility_paths = self._write_training_compatibility_files(
            root,
            result,
            evaluator,
            model_path=model_path,
            compatibility_model_path=compatibility_model_path,
            normalization_stats=normalization_stats,
            split_info=split_info,
            split_summary=split_summary,
            training_config=training_config,
            test_minerals_df=test_minerals_df,
            test_area_positions=test_area_positions,
        )

        manifest_path = root / "rebuild_manifest.json"
        manifest = {
            "model_name": result.get("model_name"),
            "model_type": result.get("model_type"),
            "model_path": str(model_path),
            "training_compatible_model_path": str(compatibility_model_path) if compatibility_model_path else "",
            "model_format": model_extension.lstrip("."),
            "h5_path": result.get("h5_path"),
            "h5_mode": result.get("dataset_summary", {}).get("h5_mode"),
            "input_channels": result.get("dataset_meta", {}).get("input_channels"),
            "image_size": result.get("dataset_meta", {}).get("image_size"),
            "num_classes": result.get("dataset_meta", {}).get("num_classes"),
            "dataset_params": dataset_params,
            "best_params": best_params,
            "workflow_config": workflow_config,
            "model_context": model_context,
            "training_config": training_config,
            "label_mapping": result.get("dataset_summary", {}).get("label_mapping"),
            "class_mapping": result.get("dataset_summary", {}).get("class_distribution"),
            "normalization_stats": normalization_stats,
            "split_info": split_info,
            "split_summary": split_summary,
            "evaluation_summary": evaluation_summary,
            "prediction_mode": "probability_map",
            "checkpoint_source": checkpoint_source,
            "thresholds": export_thresholds,
            "default_detection_threshold": best_detection_threshold,
            "best_test_threshold": best_test_threshold,
            "best_test_sr": result.get("best_test_sr"),
            "best_test_paf": result.get("best_test_paf"),
            "best_test_ei": result.get("best_test_ei"),
            **metric_protocol_fields(
                threshold_step=protocol["threshold_step"],
                distance_threshold=protocol["distance_threshold"],
            ),
            "probability_map_path": str(probability_map_path),
            "prediction_header_path": str(prediction_header_path),
            "zone_statistics_path": str(zone_statistics_path),
            "feature_schema_path": str(feature_schema_path),
            "training_summary_path": str(training_summary_path),
            "params_path": compatibility_paths.get("params_path", ""),
            "normalization_params_path": compatibility_paths.get("normalization_params_path", ""),
            "split_summary_path": compatibility_paths.get("split_summary_path", ""),
            "test_area_path": compatibility_paths.get("test_area_path", ""),
            "test_mineral_path": compatibility_paths.get("test_mineral_path", ""),
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
        src_curve = [
            {"threshold": float(item.get("threshold", 0.0) or 0.0), "sr": float(item.get("test_sr", 0.0) or 0.0)}
            for item in zone_exports
        ]
        pac_curve = [
            {"threshold": float(item.get("threshold", 0.0) or 0.0), "paf": float(item.get("test_paf", 0.0) or 0.0)}
            for item in zone_exports
        ]
        best_zone = None
        for item in zone_exports:
            try:
                if abs(float(item.get("threshold", -1.0)) - float(best_test_threshold)) <= 1e-9:
                    best_zone = item
                    break
            except (TypeError, ValueError):
                continue
        if best_zone is None and zone_exports:
            best_zone = max(zone_exports, key=lambda item: float(item.get("test_ei", 0.0) or 0.0))
        return {
            "model_artifact": asdict(model_artifact),
            "prediction_artifact": asdict(prediction_artifact),
            "model_artifact_path": str(model_path),
            "training_compatible_model_path": str(compatibility_model_path) if compatibility_model_path else "",
            "rebuild_manifest_path": str(manifest_path),
            **compatibility_paths,
            "probability_map_path": str(probability_map_path),
            "zone_statistics_path": str(zone_statistics_path),
            "sr": float((best_zone or {}).get("test_sr", 0.0) or 0.0),
            "paf": float((best_zone or {}).get("test_paf", 0.0) or 0.0),
            "ei": float((best_zone or {}).get("test_ei", 0.0) or 0.0),
            "best_test_sr": float((best_zone or {}).get("test_sr", 0.0) or 0.0),
            "best_test_paf": float((best_zone or {}).get("test_paf", 0.0) or 0.0),
            "best_test_ei": float((best_zone or {}).get("test_ei", 0.0) or 0.0),
            "best_test_threshold": best_test_threshold,
            **metric_protocol_fields(
                threshold_step=protocol["threshold_step"],
                distance_threshold=protocol["distance_threshold"],
            ),
            "src_curve": src_curve,
            "pac_curve": pac_curve,
        }

    def _save_training_model_alias(self, root: Path, model_path: Path, model_extension: str) -> Optional[Path]:
        """Create the filename expected by the training/prediction modules."""

        if not model_path.exists():
            return None
        alias_name = "model.pth" if model_extension == ".pth" else f"model{model_extension}"
        alias_path = root / alias_name
        if alias_path.resolve() != model_path.resolve():
            shutil.copy2(model_path, alias_path)
        return alias_path

    def _metric_protocol_config(self, result: Dict[str, object]) -> Dict[str, float]:
        dataset_summary = dict(result.get("dataset_summary") or {})
        dataset_params = dict(result.get("dataset_params") or {})
        protocol = dict(dataset_summary.get("evaluation_protocol") or dataset_params.get("evaluation_protocol") or {})
        threshold_step = float(protocol.get("threshold_step", result.get("threshold_step", DEFAULT_THRESHOLD_STEP)) or DEFAULT_THRESHOLD_STEP)
        distance_threshold = float(
            protocol.get("distance_threshold", result.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD))
            or DEFAULT_DISTANCE_THRESHOLD
        )
        if threshold_step <= 0 or threshold_step > 1:
            threshold_step = DEFAULT_THRESHOLD_STEP
        if distance_threshold <= 0:
            distance_threshold = DEFAULT_DISTANCE_THRESHOLD
        return {
            "threshold_step": float(threshold_step),
            "distance_threshold": float(distance_threshold),
        }

    def _resolve_test_area_positions(self, explicit_positions, result: Dict[str, object], evaluator: MineralEvaluator) -> np.ndarray:
        candidates = [
            explicit_positions,
            result.get("test_area_positions"),
            result.get("independent_test_positions"),
        ]
        dataset_summary = dict(result.get("dataset_summary") or {})
        if dataset_summary.get("test_area_positions") is not None:
            candidates.append(dataset_summary.get("test_area_positions"))
        for candidate in candidates:
            if candidate is None:
                continue
            array = np.asarray(candidate, dtype=np.float64)
            if array.ndim == 2 and array.shape[1] >= 2 and len(array) > 0:
                return array[:, :2]
        return np.empty((0, 2), dtype=np.float64)

    def _write_training_compatibility_files(
        self,
        root: Path,
        result: Dict[str, object],
        evaluator: MineralEvaluator,
        *,
        model_path: Path,
        compatibility_model_path: Optional[Path],
        normalization_stats,
        split_info,
        split_summary,
        training_config: Dict[str, object],
        test_minerals_df=None,
        test_area_positions=None,
    ) -> Dict[str, str]:
        """Write files that mirror the model-training module's result folder."""

        dataset_params = dict(result.get("dataset_params") or {})
        dataset_summary = dict(result.get("dataset_summary") or {})
        model_context = self._build_model_context_fields(result)
        resolved_prior = model_context.get("prior", training_config.get("prior", training_config.get("manual_prior")))

        protocol = self._metric_protocol_config(result)
        test_positions = self._resolve_test_area_positions(test_area_positions, result, evaluator)
        test_mineral_positions = self._extract_mineral_positions(test_minerals_df)
        test_area_path = root / "test_area.h5"
        self._write_test_area_h5(test_area_path, test_positions, test_mineral_positions)

        test_mineral_path = root / "test_minerals.txt"
        if len(test_mineral_positions) > 0:
            np.savetxt(
                test_mineral_path,
                test_mineral_positions,
                fmt="%.8f",
                delimiter="\t",
                header="x\ty",
                comments="",
            )
        else:
            test_mineral_path.write_text("x\ty\n", encoding="utf-8")

        split_summary_path = root / "split_summary.json"
        self._write_json(split_summary_path, split_summary)

        normalization_payload = {
            "mean": None if not isinstance(normalization_stats, dict) else normalization_stats.get("mean"),
            "std": None if not isinstance(normalization_stats, dict) else normalization_stats.get("std"),
            "normalization_stats": normalization_stats,
            "split_info": split_info,
            "split_summary": split_summary,
            "test_indices": np.arange(len(test_positions), dtype=np.int64),
            "test_mask": np.ones(len(test_positions), dtype=bool),
            "test_positions": test_positions,
            "test_mineral_positions": test_mineral_positions,
            **metric_protocol_fields(
                threshold_step=protocol["threshold_step"],
                distance_threshold=protocol["distance_threshold"],
            ),
            "spatial_cv_folds": dataset_summary.get("spatial_cv_folds") or result.get("spatial_cv_folds"),
            "spatial_cv_fold_count": dataset_summary.get("spatial_cv_fold_count"),
            "spatial_cv_buffer_distance": dataset_summary.get("spatial_cv_buffer_distance"),
            "spatial_cv_axis": dataset_summary.get("spatial_cv_axis"),
        }
        normalization_params_path = root / "normalization_params.pth"
        try:
            import torch

            torch.save(normalization_payload, normalization_params_path)
        except Exception:
            # Keep the export useful even in environments where torch serialization is unavailable.
            self._write_json(root / "normalization_params.json", normalization_payload)

        params_path = root / "params.json"
        params_payload = {
            "model": result.get("model_name"),
            "model_dir": str(root),
            "model_path": str(compatibility_model_path or model_path),
            "best_model_path": str(model_path),
            "auto_optimization_rank": result.get("auto_optimization_rank"),
            "selection_metric": result.get("selection_metric", "best_test_ei"),
            "primary_score_test_ei": primary_test_ei_score(result),
            "metric_protocol": METRIC_PROTOCOL,
            "threshold_strategy": THRESHOLD_STRATEGY,
            "paf_scope": PAF_SCOPE,
            "threshold_rule": THRESHOLD_RULE,
            **metric_protocol_fields(
                threshold_step=protocol["threshold_step"],
                distance_threshold=protocol["distance_threshold"],
            ),
            "best_test_ei": result.get("best_test_ei"),
            "best_test_sr": result.get("best_test_sr"),
            "best_test_paf": result.get("best_test_paf"),
            "best_test_threshold": result.get("best_test_threshold"),
            "calculated_prior": resolved_prior,
            "resolved_prior": resolved_prior,
            "normalization_params_path": str(normalization_params_path),
            "test_area_path": str(test_area_path),
            "test_mineral_path": str(test_mineral_path),
            "dataset": result.get("h5_path"),
            "h5_path": result.get("h5_path"),
            "h5_mode": dataset_summary.get("h5_mode"),
            "sample_ratio": dataset_params.get("sampling_percentage", dataset_summary.get("sampling_percentage", 1.0)),
            "sampling_percentage": dataset_params.get("sampling_percentage", dataset_summary.get("sampling_percentage", 1.0)),
            "test_size": dataset_params.get("test_split_ratio", dataset_summary.get("test_split_ratio", 0.0)),
            "split_mode": dataset_params.get("split_mode", dataset_summary.get("split_mode", "")),
            "patch_size": dataset_params.get("patch_size", dataset_summary.get("patch_size")),
            "patch_stride": dataset_params.get("patch_stride", dataset_summary.get("patch_stride")),
            "reflect_padding": dataset_params.get("reflect_padding", dataset_summary.get("reflect_padding")),
            "buffer_radius": dataset_params.get("buffer_radius", dataset_summary.get("buffer_radius")),
            "spatial_cluster_n_clusters": dataset_summary.get("spatial_cluster_n_clusters"),
            "spatial_cluster_train_ratio": dataset_summary.get("spatial_cluster_train_ratio"),
            "spatial_cv_buffer_distance": dataset_summary.get("spatial_cv_buffer_distance"),
            "spatial_random_state": dataset_summary.get("spatial_cluster_random_state"),
            "cv_folds": dataset_summary.get("spatial_cv_fold_count") or dataset_summary.get("spatial_cluster_cv_folds"),
            "batchsize": training_config.get("batch_size"),
            "batch_size": training_config.get("batch_size"),
            "epoch": training_config.get("epochs"),
            "epochs": training_config.get("epochs"),
            "stepsize": training_config.get("learning_rate"),
            "learning_rate": training_config.get("learning_rate"),
            "optimizer": training_config.get("optimizer"),
            "loss_type": model_context.get("loss_type", training_config.get("loss_type")),
            "prior_mode": model_context.get("prior_mode", training_config.get("prior_mode")),
            "beta": model_context.get("beta", training_config.get("beta")),
            "gamma": model_context.get("gamma", training_config.get("gamma")),
            "adaptive_window": model_context.get("adaptive_window", training_config.get("adaptive_window")),
            "adaptive_lambda": model_context.get("adaptive_lambda", training_config.get("adaptive_lambda")),
            "adaptive_gamma_min": model_context.get("adaptive_gamma_min", training_config.get("adaptive_gamma_min")),
            "adaptive_gamma_max": model_context.get("adaptive_gamma_max", training_config.get("adaptive_gamma_max")),
            "training_config": training_config,
            "dataset_params": dataset_params,
            "dataset_summary": dataset_summary,
            "split_summary": split_summary,
        }
        self._write_json(params_path, params_payload)

        return {
            "params_path": str(params_path),
            "normalization_params_path": str(normalization_params_path),
            "split_summary_path": str(split_summary_path),
            "test_area_path": str(test_area_path),
            "test_mineral_path": str(test_mineral_path),
        }

    def _extract_test_positions(self, evaluator: MineralEvaluator) -> np.ndarray:
        coordinates = getattr(evaluator, "coordinates", None)
        if coordinates is not None:
            return np.asarray(coordinates, dtype=np.float64)
        metadata = getattr(evaluator, "metadata", {}) or {}
        nx = int(metadata.get("nx", 0) or 0)
        ny = int(metadata.get("ny", 0) or 0)
        if nx <= 0 or ny <= 0:
            return np.empty((0, 2), dtype=np.float64)
        yy, xx = np.indices((ny, nx))
        return np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)

    def _extract_mineral_positions(self, minerals_df) -> np.ndarray:
        if minerals_df is None or len(minerals_df) == 0:
            return np.empty((0, 2), dtype=np.float64)
        frame = minerals_df.copy()
        if "x" not in frame.columns or "y" not in frame.columns:
            return np.empty((0, 2), dtype=np.float64)
        return frame[["x", "y"]].to_numpy(dtype=np.float64)

    def _write_test_area_h5(self, output_path: Path, test_positions: np.ndarray, test_mineral_positions: np.ndarray) -> None:
        import h5py

        positions = np.asarray(test_positions, dtype=np.float64).reshape(-1, 2) if len(test_positions) else np.empty((0, 2), dtype=np.float64)
        with h5py.File(output_path, "w") as handle:
            handle.attrs["metric_protocol"] = METRIC_PROTOCOL
            handle.attrs["threshold_strategy"] = THRESHOLD_STRATEGY
            handle.attrs["paf_scope"] = PAF_SCOPE
            handle.attrs["threshold_rule"] = THRESHOLD_RULE
            handle.create_dataset("positions", data=positions)
            handle.create_dataset(
                "test_mineral_positions",
                data=np.asarray(test_mineral_positions, dtype=np.float64).reshape(-1, 2)
                if len(test_mineral_positions)
                else np.empty((0, 2), dtype=np.float64),
            )

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

        manifest_dataset_params = dict(manifest.get("dataset_params", {}) or {})
        rebuild_patch_size = int(manifest_dataset_params.get("patch_size") or dataset_meta["image_size"])
        rebuild_patch_stride = int(manifest_dataset_params.get("patch_stride") or rebuild_patch_size)
        evaluator = MineralEvaluator(
            str(manifest["h5_path"]),
            buffer_radius=float(manifest_dataset_params.get("buffer_radius", 500)),
            prediction_patch_size=rebuild_patch_size,
            prediction_patch_stride=rebuild_patch_stride,
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
                "best_test_threshold": manifest.get("best_test_threshold"),
                "best_test_sr": manifest.get("best_test_sr"),
                "best_test_paf": manifest.get("best_test_paf"),
                "best_test_ei": manifest.get("best_test_ei"),
                **metric_protocol_fields(
                    threshold_step=float(manifest.get("threshold_step", DEFAULT_THRESHOLD_STEP) or DEFAULT_THRESHOLD_STEP),
                    distance_threshold=float(manifest.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD) or DEFAULT_DISTANCE_THRESHOLD),
                ),
            },
        )
        prediction_header_path = rebuild_root / "prediction_header.json"
        self._write_json(prediction_header_path, prediction_header)
        preview_image_path = rebuild_root / "probability_map.png"
        self._save_probability_preview(preview_image_path, probability_map, metadata=evaluator.metadata)

        zones_dir = rebuild_root / "zones"
        zones_dir.mkdir(parents=True, exist_ok=True)
        test_area_positions = self._load_test_area_positions_from_path(manifest.get("test_area_path"))
        test_minerals_df = self._load_test_minerals_from_path(manifest.get("test_mineral_path"))
        zone_exports = self._export_threshold_zones(
            zones_dir,
            probability_map,
            evaluator,
            val_minerals_df=None,
            test_minerals_df=test_minerals_df,
            test_area_positions=test_area_positions,
            threshold_step=float(manifest.get("threshold_step", DEFAULT_THRESHOLD_STEP) or DEFAULT_THRESHOLD_STEP),
            distance_threshold=float(manifest.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD) or DEFAULT_DISTANCE_THRESHOLD),
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

    def _load_test_area_positions_from_path(self, path_like) -> Optional[np.ndarray]:
        if not path_like:
            return None
        path = Path(str(path_like or ""))
        if not path.exists():
            return None
        suffix = path.suffix.lower()
        try:
            if suffix in {".h5", ".hdf5"}:
                import h5py

                with h5py.File(path, "r") as handle:
                    key = next((name for name in ("positions", "coordinates", "coords", "test_positions") if name in handle), None)
                    if key is None:
                        return None
                    data = np.asarray(handle[key][:], dtype=np.float64)
            elif suffix == ".npy":
                data = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
            elif suffix == ".npz":
                payload = np.load(path, allow_pickle=False)
                key = next((name for name in ("positions", "coordinates", "coords", "test_positions") if name in payload), None)
                data = np.asarray(payload[key], dtype=np.float64) if key is not None else np.empty((0, 2), dtype=np.float64)
            else:
                table = pd.read_csv(path, sep=None, engine="python")
                columns = {str(col).strip().lower(): col for col in table.columns}
                x_col = next((columns[name] for name in ("x", "coord_x", "east", "easting") if name in columns), table.columns[0])
                y_col = next((columns[name] for name in ("y", "coord_y", "north", "northing") if name in columns), table.columns[1])
                data = table[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float64)
            if data.ndim == 2 and data.shape[1] >= 2 and len(data) > 0:
                return data[:, :2]
        except Exception:
            return None
        return None

    def _load_test_minerals_from_path(self, path_like):
        if not path_like:
            return None
        path = Path(str(path_like or ""))
        if not path.exists():
            return None
        try:
            table = pd.read_csv(path, sep=None, engine="python")
            if len(table.columns) < 2:
                table = pd.read_csv(path, sep=r"[\s,]+", header=None, engine="python")
            columns = {str(col).strip().lower(): col for col in table.columns}
            x_col = next((columns[name] for name in ("x", "coord_x", "east", "easting") if name in columns), table.columns[0])
            y_col = next((columns[name] for name in ("y", "coord_y", "north", "northing") if name in columns), table.columns[1])
            frame = table[[x_col, y_col]].copy()
            frame.columns = ["x", "y"]
            frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
            frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
            return frame.dropna(subset=["x", "y"]).reset_index(drop=True)
        except Exception:
            return None

    def _export_threshold_zones(
        self,
        zones_dir: Path,
        probability_map: np.ndarray,
        evaluator: MineralEvaluator,
        *,
        val_minerals_df,
        test_minerals_df,
        test_area_positions=None,
        threshold_step: float = DEFAULT_THRESHOLD_STEP,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        thresholds: Optional[Iterable[float]] = None,
    ) -> List[Dict[str, object]]:
        zone_exports: List[Dict[str, object]] = []
        for threshold in thresholds or self.thresholds:
            threshold_value = float(threshold)
            threshold_tag = f"{threshold_value:.2f}".replace(".", "_")
            mask = np.asarray(probability_map, dtype=np.float64) > threshold_value
            stem = zones_dir / f"threshold_{threshold_tag}"
            mask_path = stem.with_suffix(".npy")
            preview_path = stem.with_suffix(".png")
            np.save(mask_path, mask.astype(np.uint8))
            self._save_zone_preview(preview_path, mask, metadata=evaluator.metadata)
            val_detection = evaluator.evaluate_probability_map(probability_map, val_minerals_df, threshold=float(threshold))
            test_detection = None
            test_metric_row = None
            if test_minerals_df is not None and len(test_minerals_df) > 0 and test_area_positions is not None and len(test_area_positions) > 0:
                try:
                    metric_bundle = evaluate_independent_test_metrics(
                        probability_map,
                        evaluator.metadata,
                        test_minerals_df,
                        test_positions=test_area_positions,
                        fixed_threshold=float(threshold),
                        threshold_step=float(threshold_step or DEFAULT_THRESHOLD_STEP),
                        distance_threshold=float(distance_threshold or DEFAULT_DISTANCE_THRESHOLD),
                    )
                    test_metric_row = dict(metric_bundle.get("best") or {})
                    test_detection = detection_from_metric_row(test_metric_row)
                except Exception:
                    test_detection = None
            if test_detection is None:
                test_detection = evaluator.evaluate_probability_map(probability_map, test_minerals_df, threshold=float(threshold))
            stats = self._compute_zone_stats(
                mask,
                evaluator.metadata,
                threshold,
                val_detection,
                test_detection,
                test_metric_row=test_metric_row,
                threshold_step=float(threshold_step or DEFAULT_THRESHOLD_STEP),
                distance_threshold=float(distance_threshold or DEFAULT_DISTANCE_THRESHOLD),
            )
            stats_path = stem.with_name(stem.name + "_stats.json")
            self._write_json(stats_path, stats)
            stats["mask_path"] = str(mask_path)
            stats["preview_path"] = str(preview_path)
            stats["stats_path"] = str(stats_path)
            zone_exports.append(stats)
        return zone_exports

    def _build_prediction_header(self, evaluator: MineralEvaluator, result: Dict[str, object]) -> Dict[str, object]:
        metadata = {key: self._to_jsonable(value) for key, value in evaluator.metadata.items()}
        protocol = self._metric_protocol_config(result)
        return {
            "metadata": metadata,
            "dataset_params": self._to_jsonable(result.get("dataset_params", {})),
            "best_detection_threshold": self._to_jsonable(result.get("best_detection_threshold", DETECTION_THRESHOLD)),
            "best_test_threshold": self._to_jsonable(result.get("best_test_threshold")),
            "best_test_sr": self._to_jsonable(result.get("best_test_sr")),
            "best_test_paf": self._to_jsonable(result.get("best_test_paf")),
            "best_test_ei": self._to_jsonable(result.get("best_test_ei")),
            **metric_protocol_fields(
                threshold_step=protocol["threshold_step"],
                distance_threshold=protocol["distance_threshold"],
            ),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _compute_zone_stats(
        self,
        mask: np.ndarray,
        metadata: Dict[str, object],
        threshold: float,
        val_detection: Optional[Dict[str, object]],
        test_detection: Optional[Dict[str, object]],
        *,
        test_metric_row: Optional[Dict[str, object]] = None,
        threshold_step: float = DEFAULT_THRESHOLD_STEP,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
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

        test_metric_row = dict(test_metric_row or {})
        test_sr = float(test_metric_row.get("test_sr", (test_detection or {}).get("detection_rate", 0.0)) or 0.0)
        if test_metric_row:
            test_paf = float(test_metric_row.get("test_paf", 0.0) or 0.0)
            test_ei = float(test_metric_row.get("test_ei", (test_sr / test_paf if test_paf > 0 else 0.0)) or 0.0)
        else:
            test_paf = None
            test_ei = None
        test_area_count = int(test_metric_row.get("test_area_count", 0) or 0)
        test_high_count = int(test_metric_row.get("high_potential_count", 0) or 0)

        return {
            "threshold": float(threshold),
            **metric_protocol_fields(
                threshold_step=float(threshold_step or DEFAULT_THRESHOLD_STEP),
                distance_threshold=float(distance_threshold or DEFAULT_DISTANCE_THRESHOLD),
            ),
            "zone_count": int(self._count_connected_components(mask)),
            "pixel_count": positive_pixels,
            "total_pixel_count": total_pixels,
            "area_ratio": area_ratio,
            "area_estimate": float(positive_pixels * pixel_area),
            "test_area_pixel_count": test_area_count,
            "test_high_potential_count": test_high_count,
            "val_mineral_count": int((val_detection or {}).get("mineral_count", 0)),
            "val_detected_count": int((val_detection or {}).get("detected_count", 0)),
            "val_detection_rate": float((val_detection or {}).get("detection_rate", 0.0)),
            "test_mineral_count": int((test_detection or {}).get("mineral_count", 0)),
            "test_detected_count": int((test_detection or {}).get("detected_count", 0)),
            "test_detection_rate": test_sr,
            "val_sr": float((val_detection or {}).get("detection_rate", 0.0)),
            "val_paf": area_ratio,
            "val_ei": float((float((val_detection or {}).get("detection_rate", 0.0)) / area_ratio) if area_ratio > 0 else 0.0),
            "test_sr": test_sr,
            "test_paf": test_paf,
            "test_ei": test_ei,
        }

    def _leaderboard_row(self, result: Dict[str, object]) -> Dict[str, object]:
        dataset_params = dict(result.get("dataset_params") or {})
        split_summary = self._build_split_summary_fields(result)
        evaluation_summary = self._build_evaluation_summary_fields(result)
        context_fields = self._build_model_context_fields(result)
        return {
            "model_name": result.get("model_name", ""),
            "model_family": context_fields.get("model_family", ""),
            "input_kind": context_fields.get("input_kind", ""),
            "label_mode": context_fields.get("label_mode", ""),
            "training_mode": context_fields.get("training_mode", ""),
            "is_pu_model": context_fields.get("is_pu_model", False),
            "loss_type": context_fields.get("loss_type", ""),
            "prior_mode": context_fields.get("prior_mode", ""),
            "prior": context_fields.get("prior", ""),
            "beta": context_fields.get("beta", ""),
            "gamma": context_fields.get("gamma", ""),
            "adaptive_window": context_fields.get("adaptive_window", ""),
            "adaptive_lambda": context_fields.get("adaptive_lambda", ""),
            "adaptive_gamma_min": context_fields.get("adaptive_gamma_min", ""),
            "adaptive_gamma_max": context_fields.get("adaptive_gamma_max", ""),
            "selection_strategy": result.get("selection_strategy", "composite"),
            "metric_protocol": result.get("metric_protocol", METRIC_PROTOCOL),
            "threshold_strategy": result.get("threshold_strategy", THRESHOLD_STRATEGY),
            "paf_scope": result.get("paf_scope", PAF_SCOPE),
            "threshold_rule": result.get("threshold_rule", THRESHOLD_RULE),
            "is_best_by_composite": bool(result.get("is_best_by_composite", False)),
            "is_best_by_ei": bool(result.get("is_best_by_ei", False)),
            "primary_score_test_ei": primary_test_ei_score(result),
            "learning_rate": context_fields.get("learning_rate", ""),
            "batch_size": context_fields.get("batch_size", ""),
            "optimizer": context_fields.get("optimizer", ""),
            "split_mode": split_summary.get("split_mode", ""),
            "train_sample_count": split_summary.get("train_sample_count", 0),
            "val_sample_count": split_summary.get("val_sample_count", 0),
            "test_sample_count": split_summary.get("test_sample_count", 0),
            "train_mineral_count": split_summary.get("train_mineral_count", 0),
            "val_mineral_count": split_summary.get("val_mineral_count", 0),
            "test_mineral_count": split_summary.get("test_mineral_count", 0),
            "spatial_cluster_active": split_summary.get("spatial_cluster_active", False),
            "spatial_cluster_n_clusters": split_summary.get("spatial_cluster_n_clusters", 0),
            "spatial_cluster_train_ratio": split_summary.get("spatial_cluster_train_ratio", 0.0),
            "spatial_cluster_cv_folds": split_summary.get("spatial_cluster_cv_folds", 0),
            "spatial_cluster_random_state": split_summary.get("spatial_cluster_random_state", 0),
            "spatial_cv_enabled": split_summary.get("spatial_cv_enabled", False),
            "spatial_cv_fold_count": split_summary.get("spatial_cv_fold_count", 0),
            "spatial_cv_axis": split_summary.get("spatial_cv_axis", ""),
            "spatial_cv_buffer_distance": split_summary.get("spatial_cv_buffer_distance", 0.0),
            "spatial_cv_selected_fold": split_summary.get("spatial_cv_selected_fold", 0),
            "independent_test_sample_count": evaluation_summary.get("independent_test_sample_count", 0),
            "independent_test_mineral_count": evaluation_summary.get("independent_test_mineral_count", 0),
            "independent_test_accuracy": evaluation_summary.get("independent_test_accuracy"),
            "independent_test_f1": evaluation_summary.get("independent_test_f1"),
            "independent_test_mineral_detection_rate": evaluation_summary.get("independent_test_mineral_detection_rate"),
            "composite_score": result.get("composite_score"),
            "val_accuracy": result.get("val_accuracy"),
            "val_f1": result.get("val_f1"),
            "val_sr": result.get("val_sr"),
            "val_paf": result.get("val_paf"),
            "val_ei": result.get("val_ei"),
            "cv_ei_mean": result.get("cv_ei_mean"),
            "cv_ei_std": result.get("cv_ei_std"),
            "val_mineral_detection_rate": result.get("val_mineral_detection_rate"),
            "test_accuracy": result.get("test_accuracy"),
            "test_f1": result.get("test_f1"),
            "test_sr": result.get("test_sr"),
            "test_paf": result.get("test_paf"),
            "test_ei": result.get("test_ei"),
            "best_test_threshold": result.get("best_test_threshold"),
            "best_test_sr": result.get("best_test_sr"),
            "best_test_paf": result.get("best_test_paf"),
            "best_test_ei": result.get("best_test_ei"),
            "src_curve_points": len(result.get("src_curve") or []),
            "pac_curve_points": len(result.get("pac_curve") or []),
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

    def _image_extent(self, metadata: Optional[Dict[str, object]]) -> Optional[tuple]:
        meta = metadata or {}
        try:
            return (
                float(meta["x_min"]),
                float(meta["x_max"]),
                float(meta["y_min"]),
                float(meta["y_max"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _preview_figure_size(self, array: np.ndarray) -> tuple:
        height, width = np.asarray(array).shape[:2]
        aspect = width / max(height, 1)
        fig_width = min(max(6.0, 4.5 * aspect), 12.0)
        fig_height = min(max(3.0, fig_width / max(aspect, 0.25)), 8.0)
        return fig_width, fig_height

    def _save_probability_preview(
        self,
        output_path: Path,
        probability_map: np.ndarray,
        *,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=self._preview_figure_size(probability_map))
        image = ax.imshow(
            probability_map,
            cmap="viridis",
            origin="upper",
            extent=self._image_extent(metadata),
            aspect="equal",
        )
        fig.colorbar(image, ax=ax, label="Probability", fraction=0.046, pad=0.04)
        ax.set_title("Probability Map")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_zone_preview(
        self,
        output_path: Path,
        zone_mask: np.ndarray,
        *,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=self._preview_figure_size(zone_mask))
        image = ax.imshow(
            zone_mask.astype(np.uint8),
            cmap="magma",
            origin="upper",
            extent=self._image_extent(metadata),
            aspect="equal",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Prediction Zone")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

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
        spec = SYSTEM_MODEL_SPECS.get(model_name)
        if spec is not None:
            return str(spec.get("family")) == "neural"
        return model_name.startswith("CNN") or model_name.startswith("ResNet")
