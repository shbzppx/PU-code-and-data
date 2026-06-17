"""AutoML engine for classic ML and deep learning comparison."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.metrics import precision_recall_fscore_support

from .analysis_utils import (
    DETECTION_THRESHOLD,
    build_improvement_advice,
    compute_composite_score,
    compute_sr_paf_ei,
    detect_search_boundary_hits,
    get_detection_rate,
    primary_test_ei_score,
    primary_test_ei_sort_key,
    resolve_composite_formula,
)
from .data_views import prepare_model_views, prepare_training_view
from .mineral_evaluator import MineralEvaluator
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
    threshold_candidates,
)
from .optuna_optimizer import OptunaOptimizer


class AutoMLEngine(QObject):
    """Manage AutoML runs across registered model wrappers."""

    model_started = pyqtSignal(str)
    model_progress = pyqtSignal(str, int, int)
    model_completed = pyqtSignal(str, dict)
    all_completed = pyqtSignal(list)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.models = {}
        self.results = []
        self.is_running = False
        self.current_wrapper = None

    def _threshold_candidates(self):
        return [float(value) for value in threshold_candidates(step=DEFAULT_THRESHOLD_STEP)]

    def _metric_protocol_config(self, dataset_bundle):
        summary = dict(getattr(dataset_bundle, "dataset_summary", {}) or {})
        protocol = dict(summary.get("evaluation_protocol") or {})
        threshold_step = float(protocol.get("threshold_step", DEFAULT_THRESHOLD_STEP) or DEFAULT_THRESHOLD_STEP)
        distance_threshold = float(
            protocol.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD)
            or DEFAULT_DISTANCE_THRESHOLD
        )
        return threshold_step, distance_threshold

    def _select_best_threshold(self, evaluator, probability_map, mineral_df):
        if evaluator is None or mineral_df is None or len(mineral_df) == 0:
            return None, None, []

        threshold_history = []
        best_threshold = None
        best_detection = None
        best_score = (-1.0, -1.0, -1.0)
        for threshold in self._threshold_candidates():
            detection = evaluator.evaluate_probability_map(probability_map, mineral_df, threshold=threshold)
            detection_rate = float((detection or {}).get("detection_rate", 0.0))
            avg_probability = float((detection or {}).get("avg_probability", 0.0))
            threshold_history.append(
                {
                    "threshold": float(threshold),
                    "detection_rate": detection_rate,
                    "avg_probability": avg_probability,
                }
            )
            candidate_score = (detection_rate, avg_probability, float(threshold))
            if candidate_score > best_score:
                best_score = candidate_score
                best_threshold = float(threshold)
                best_detection = detection
        return best_threshold, best_detection, threshold_history

    def _compute_test_threshold_curves(
        self,
        evaluator,
        probability_map,
        test_minerals_df,
        *,
        test_area_positions=None,
        test_area_mask=None,
        threshold_step=DEFAULT_THRESHOLD_STEP,
        distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
    ):
        if evaluator is None or probability_map is None or test_minerals_df is None or len(test_minerals_df) == 0:
            return {
                "best_test_threshold": None,
                "best_test_detection": None,
                "threshold_curve": [],
                "src_curve": [],
                "pac_curve": [],
            }

        try:
            metrics = evaluate_independent_test_metrics(
                probability_map,
                getattr(evaluator, "metadata", {}) or {},
                test_minerals_df,
                test_area_mask=test_area_mask,
                test_positions=test_area_positions,
                threshold_step=float(threshold_step or DEFAULT_THRESHOLD_STEP),
                distance_threshold=float(distance_threshold or DEFAULT_DISTANCE_THRESHOLD),
            )
        except Exception:
            return {
                "best_test_threshold": None,
                "best_test_detection": None,
                "threshold_curve": [],
                "src_curve": [],
                "pac_curve": [],
            }

        threshold_curve = list(metrics.get("threshold_curve") or [])
        src_curve = list(metrics.get("src_curve") or [])
        pac_curve = list(metrics.get("pac_curve") or [])
        best_row = dict(metrics.get("best") or {})
        best_detection = detection_from_metric_row(best_row)
        return {
            "best_test_threshold": None if best_row is None else float(best_row["threshold"]),
            "best_test_detection": best_detection,
            "best_test_sr": None if best_row is None else float(best_row["test_sr"]),
            "best_test_paf": None if best_row is None else float(best_row["test_paf"]),
            "best_test_ei": None if best_row is None else float(best_row["test_ei"]),
            "threshold_curve": threshold_curve,
            "src_curve": src_curve,
            "pac_curve": pac_curve,
        }

    @staticmethod
    def _resolve_augmentation_config(dataset_bundle, wrapper=None):
        summary = dict(getattr(dataset_bundle, "dataset_summary", {}) or {})
        wrapper_training = dict(getattr(wrapper, "training_config", {}) or {})
        enabled = summary.get("augmentation_enabled", wrapper_training.get("augmentation_enabled", False))
        noise_std = summary.get("augmentation_noise_std", wrapper_training.get("augmentation_noise_std", 0.01))
        return {
            "augmentation_enabled": bool(enabled),
            "augmentation_noise_std": float(noise_std if noise_std not in (None, "") else 0.01),
        }

    def register_model(self, name, wrapper):
        self.models[name] = wrapper
        self.log_message.emit(f"已注册模型: {name}")

    def clear_models(self):
        self.models.clear()
        self.results.clear()

    def run_automl(self, dataset_bundle, n_trials=50):
        self.is_running = True
        self.results = []
        use_spatial_cv = self._has_spatial_cv(dataset_bundle)

        evaluator = None
        if dataset_bundle.h5_path and (
            (dataset_bundle.val_minerals_df is not None and len(dataset_bundle.val_minerals_df) > 0)
            or (dataset_bundle.test_minerals_df is not None and len(dataset_bundle.test_minerals_df) > 0)
        ):
            evaluator = MineralEvaluator(
                dataset_bundle.h5_path,
                buffer_radius=dataset_bundle.dataset_summary.get("buffer_radius", 500),
                prediction_patch_size=dataset_bundle.dataset_summary.get("patch_size"),
                prediction_patch_stride=dataset_bundle.dataset_summary.get("patch_stride"),
            )

        for model_name, wrapper in self.models.items():
            if not self.is_running:
                break

            self.current_wrapper = wrapper
            self.model_started.emit(model_name)
            self.log_message.emit(f"\n开始优化 {model_name}")
            train_data, val_data, test_data = self._select_split_views(wrapper, dataset_bundle)
            start_time = time.time()

            optimizer = OptunaOptimizer(
                wrapper,
                train_data,
                val_data,
                score_fn=lambda active_wrapper, params, history, trial=None: self._score_trial(
                    active_wrapper,
                    params,
                    history,
                    dataset_bundle,
                    evaluator,
                    train_data=train_data,
                    val_data=val_data,
                    use_spatial_cv=use_spatial_cv,
                    trial=trial,
                ),
                train_on_trial=False,
            )
            optimizer.trial_completed.connect(
                lambda trial_num, params, score, name=model_name: self.model_progress.emit(
                    name,
                    trial_num + 1,
                    n_trials,
                )
            )

            try:
                best_params, best_score = optimizer.optimize(n_trials=n_trials)
                if not self.is_running:
                    break

                best_trial_record = self._get_best_trial_record(optimizer.get_optimization_history())
                augmentation_config = self._resolve_augmentation_config(dataset_bundle, wrapper)
                final_train_data = prepare_training_view(
                    wrapper,
                    self._get_final_fit_view(dataset_bundle),
                    supervised_train_ratio=1.0,
                    augmentation_enabled=augmentation_config["augmentation_enabled"],
                    augmentation_noise_std=augmentation_config["augmentation_noise_std"],
                )
                final_history = wrapper.train(final_train_data, None, **best_params)
                val_accuracy = self._metric_or_default((best_trial_record or {}).get("val_accuracy"), 0.0)
                val_precision = self._metric_or_default((best_trial_record or {}).get("val_precision"), 0.0)
                val_recall = self._metric_or_default((best_trial_record or {}).get("val_recall"), 0.0)
                val_f1 = self._metric_or_default((best_trial_record or {}).get("val_f1"), 0.0)
                val_sr = self._metric_or_default((best_trial_record or {}).get("val_sr"), 0.0)
                val_paf = self._metric_or_default((best_trial_record or {}).get("val_paf"), 0.0)
                val_ei = self._metric_or_default((best_trial_record or {}).get("val_ei"), 0.0)
                cv_ei_mean = self._metric_or_default((best_trial_record or {}).get("cv_ei_mean"), 0.0)
                cv_ei_std = self._metric_or_default((best_trial_record or {}).get("cv_ei_std"), 0.0)
                val_detection = (best_trial_record or {}).get("val_mineral_detection")
                test_metrics = self._evaluate_classification(wrapper, test_data, prefix="test")
                training_time = time.time() - start_time
                train_accuracy = self._metric_or_default(final_history.get("train_acc"), 0.0)

                test_detection = None
                best_detection_threshold = (best_trial_record or {}).get("best_detection_threshold")
                best_val_detection_threshold = best_detection_threshold
                threshold_search = (best_trial_record or {}).get("threshold_search", [])
                test_threshold_curve = []
                src_curve = []
                pac_curve = []
                best_test_threshold = None
                best_test_sr = None
                best_test_paf = None
                best_test_ei = None
                threshold_step, distance_threshold = self._metric_protocol_config(dataset_bundle)

                if evaluator is not None and dataset_bundle.test_minerals_df is not None and len(dataset_bundle.test_minerals_df) > 0:
                    try:
                        probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
                        curve_bundle = self._compute_test_threshold_curves(
                            evaluator,
                            probability_map,
                            dataset_bundle.test_minerals_df,
                            test_area_positions=getattr(dataset_bundle, "test_area_positions", None),
                            threshold_step=threshold_step,
                            distance_threshold=distance_threshold,
                        )
                        test_threshold_curve = list(curve_bundle.get("threshold_curve") or [])
                        src_curve = list(curve_bundle.get("src_curve") or [])
                        pac_curve = list(curve_bundle.get("pac_curve") or [])
                        best_test_threshold = curve_bundle.get("best_test_threshold")
                        best_test_sr = curve_bundle.get("best_test_sr")
                        best_test_paf = curve_bundle.get("best_test_paf")
                        best_test_ei = curve_bundle.get("best_test_ei")
                        if best_test_threshold is not None:
                            best_detection_threshold = float(best_test_threshold)
                            test_detection = curve_bundle.get("best_test_detection")
                        else:
                            threshold_for_test = float(best_detection_threshold or DETECTION_THRESHOLD)
                            test_detection = evaluator.evaluate_probability_map(
                                probability_map,
                                dataset_bundle.test_minerals_df,
                                threshold=threshold_for_test,
                            )
                    except Exception as exc:
                        self.log_message.emit(f"{model_name} test mineral evaluation skipped: {exc}")

                val_detection_rate = get_detection_rate(val_detection)
                if val_detection_rate is None:
                    val_detection_rate = (best_trial_record or {}).get("val_mineral_detection_rate")
                test_detection_rate = get_detection_rate(test_detection)
                scoring_detection = self._first_available_metric(val_detection_rate, val_recall, default=0.0)
                composite_score = (
                    float(best_test_ei)
                    if best_test_ei is not None
                    else float((best_trial_record or {}).get("value", best_score))
                    if use_spatial_cv
                    else compute_composite_score(
                        scoring_detection,
                        val_paf,
                        None,
                    )
                )

                result = {
                    "model_name": model_name,
                    "model_type": model_name,
                    "model_context": self._build_model_context(wrapper),
                    "dataset_meta": {
                        **dict(getattr(dataset_bundle, "dataset_meta", {}) or {}),
                        "input_kind": getattr(wrapper, "input_kind", "vector"),
                        "label_mode": getattr(wrapper, "label_mode", "binary"),
                    },
                    "best_params": best_params,
                    "best_score": float(best_score),
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_sr": val_sr,
                    "val_paf": val_paf,
                    "val_ei": val_ei,
                    "cv_ei_mean": cv_ei_mean,
                    "cv_ei_std": cv_ei_std,
                    "cv_fold_metrics": list((best_trial_record or {}).get("cv_fold_metrics") or []),
                    "cv_fold_count": int((best_trial_record or {}).get("cv_fold_count", 0) or 0),
                    "test_accuracy": test_metrics.get("test_accuracy"),
                    "test_precision": test_metrics.get("test_precision"),
                    "test_recall": test_metrics.get("test_recall"),
                    "test_f1": test_metrics.get("test_f1"),
                    "test_classification_sr": test_metrics.get("test_sr"),
                    "test_classification_paf": test_metrics.get("test_paf"),
                    "test_classification_ei": test_metrics.get("test_ei"),
                    "test_sr": best_test_sr if best_test_sr is not None else test_metrics.get("test_sr"),
                    "test_paf": best_test_paf if best_test_paf is not None else test_metrics.get("test_paf"),
                    "test_ei": best_test_ei if best_test_ei is not None else test_metrics.get("test_ei"),
                    "training_time": training_time,
                    "optimization_history": optimizer.get_optimization_history(),
                    "dataset_summary": dict(dataset_bundle.dataset_summary or {}),
                    "training_config": dict(getattr(wrapper, "training_config", {}) or best_params),
                    "val_mineral_detection": val_detection,
                    "test_mineral_detection": test_detection,
                    "val_mineral_detection_rate": val_detection_rate,
                    "test_mineral_detection_rate": test_detection_rate,
                    "best_detection_threshold": best_detection_threshold,
                    "best_val_detection_threshold": best_val_detection_threshold,
                    "best_test_threshold": best_test_threshold,
                    "best_test_sr": best_test_sr,
                    "best_test_paf": best_test_paf,
                    "best_test_ei": best_test_ei,
                    "test_threshold_curve": test_threshold_curve,
                    "src_curve": src_curve,
                    "pac_curve": pac_curve,
                    "threshold_search": threshold_search,
                    "primary_score": primary_test_ei_score(
                        {
                            "best_test_ei": best_test_ei,
                            "test_ei": test_metrics.get("test_ei"),
                            "cv_ei_mean": cv_ei_mean,
                            "val_ei": val_ei,
                            "composite_score": composite_score,
                        }
                    ),
                    "selection_metric": "best_test_ei",
                    "metric_protocol": METRIC_PROTOCOL,
                    "threshold_strategy": THRESHOLD_STRATEGY,
                    "paf_scope": PAF_SCOPE,
                    "threshold_rule": THRESHOLD_RULE,
                    **metric_protocol_fields(
                        threshold_step=threshold_step,
                        distance_threshold=distance_threshold,
                    ),
                    "composite_formula": "independent_test_ei",
                    "composite_score": composite_score,
                    # Keep the final trained wrapper in memory so export can reuse the exact evaluated model.
                    "_trained_wrapper": wrapper,
                    "results": {
                        "train_acc": train_accuracy,
                        "val_acc": val_accuracy,
                    },
                }
                result["search_boundary_hits"] = detect_search_boundary_hits(
                    wrapper.get_param_space(),
                    best_params,
                )
                result["improvement_advice"] = build_improvement_advice(
                    result,
                    param_space=wrapper.get_param_space(),
                    best_params=best_params,
                    optimization_history=result["optimization_history"],
                    n_trials=n_trials,
                )

                self.results.append(result)
                self.model_completed.emit(model_name, result)
            except Exception as exc:
                message = f"{model_name} AutoML 失败: {exc}"
                self.log_message.emit(message)
                self.error_occurred.emit(message)

        self.current_wrapper = None
        self.is_running = False
        for item in self.results:
            item["primary_score"] = primary_test_ei_score(item)
            item["selection_metric"] = "best_test_ei"
        self.results.sort(key=primary_test_ei_sort_key)
        best_by_composite = self._best_result_by("composite_score")
        best_by_ei = self.results[0] if self.results else None
        for item in self.results:
            item["selection_strategy"] = "test_ei"
            item["is_best_by_composite"] = bool(item is best_by_composite)
            item["is_best_by_ei"] = bool(item is best_by_ei)
        self.all_completed.emit(self.results)
        return self.results

    def _has_spatial_cv(self, dataset_bundle):
        spatial_cv = getattr(dataset_bundle, "spatial_cv_splits", None)
        return bool(spatial_cv and spatial_cv.get("folds"))

    @staticmethod
    def _build_model_context(wrapper) -> dict:
        return {
            "model_name": getattr(wrapper, "model_name", getattr(wrapper, "model_type", "")),
            "model_family": getattr(wrapper, "model_family", "unknown"),
            "input_kind": getattr(wrapper, "input_kind", "vector"),
            "label_mode": getattr(wrapper, "label_mode", "binary"),
            "training_mode": getattr(wrapper, "training_mode", "supervised"),
            "is_pu_model": bool(getattr(wrapper, "is_pu_model", False)),
        }

    def _get_best_trial_record(self, history):
        records = [item for item in (history or []) if item.get("status") != "failed"]
        if not records:
            return None
        return max(records, key=lambda item: float(item.get("value", 0.0)))

    def _get_final_fit_view(self, dataset_bundle):
        if getattr(dataset_bundle, "dev_data_array", None) is not None:
            return dataset_bundle.dev_data_array
        if dataset_bundle.train_data_array is not None and dataset_bundle.val_data_array is not None:
            train_x, train_y = dataset_bundle.train_data_array
            val_x, val_y = dataset_bundle.val_data_array
            if train_x is not None and val_x is not None:
                return (
                    np.concatenate([np.asarray(train_x), np.asarray(val_x)], axis=0),
                    np.concatenate([np.asarray(train_y), np.asarray(val_y)], axis=0),
                )
        return dataset_bundle.train_data_array

    @staticmethod
    def _metric_or_default(value, default=0.0):
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _first_available_metric(*values, default=0.0):
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    @staticmethod
    def _find_xy_columns(frame):
        normalized = {str(column).strip().lower(): column for column in frame.columns}
        x_aliases = ["x", "coord_x", "point_x", "east", "easting", "东坐标", "横坐标"]
        y_aliases = ["y", "coord_y", "point_y", "north", "northing", "北坐标", "纵坐标"]
        x_column = next((normalized.get(alias) for alias in x_aliases if normalized.get(alias) is not None), None)
        y_column = next((normalized.get(alias) for alias in y_aliases if normalized.get(alias) is not None), None)
        if x_column is None or y_column is None:
            raise KeyError("Mineral file must contain X/Y coordinate columns.")
        return x_column, y_column

    def _development_minerals_frame(self, dataset_bundle):
        frames = []
        for frame in (getattr(dataset_bundle, "train_minerals_df", None), getattr(dataset_bundle, "val_minerals_df", None)):
            if frame is not None and len(frame) > 0:
                frames.append(frame.reset_index(drop=True))
        if not frames:
            return pd.DataFrame(columns=["x", "y"])
        return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    def _spatial_cv_fold_minerals(self, minerals_df, spatial_cv, fold_index):
        if minerals_df is None or len(minerals_df) == 0:
            return pd.DataFrame(columns=[] if minerals_df is None else minerals_df.columns)

        edges = np.asarray(spatial_cv.get("edges") or [], dtype=np.float64)
        if len(edges) < 2:
            return minerals_df.iloc[0:0].copy()

        x_column, y_column = self._find_xy_columns(minerals_df)
        coords = minerals_df[[x_column, y_column]].to_numpy(dtype=np.float64)
        axis = int(spatial_cv.get("axis", 0) or 0)
        axis_values = coords[:, axis]
        boundaries = edges[1:-1]
        block_ids = np.searchsorted(boundaries, axis_values, side="right")
        block_ids = np.clip(block_ids, 0, len(edges) - 2)

        buffer_distance = float(spatial_cv.get("buffer_distance", 0.0) or 0.0)
        gray_mask = np.zeros(len(minerals_df), dtype=bool)
        for boundary in boundaries:
            gray_mask |= np.abs(axis_values - float(boundary)) <= buffer_distance

        val_mask = (block_ids == int(fold_index)) & (~gray_mask)
        return minerals_df.loc[val_mask].reset_index(drop=True)

    def _score_trial(
        self,
        wrapper,
        params,
        history,
        dataset_bundle,
        evaluator,
        *,
        train_data=None,
        val_data=None,
        use_spatial_cv=False,
        trial=None,
    ):
        del history, trial
        if (
            evaluator is not None
            and dataset_bundle is not None
            and getattr(dataset_bundle, "test_minerals_df", None) is not None
            and len(dataset_bundle.test_minerals_df) > 0
            and getattr(dataset_bundle, "test_area_positions", None) is not None
            and len(dataset_bundle.test_area_positions) > 0
        ):
            return self._score_trial_single_split(wrapper, params, train_data, val_data, evaluator, dataset_bundle)
        if use_spatial_cv and self._has_spatial_cv(dataset_bundle):
            return self._score_trial_spatial_cv(wrapper, params, dataset_bundle)
        return self._score_trial_single_split(wrapper, params, train_data, val_data, evaluator, dataset_bundle)

    def _score_trial_single_split(self, wrapper, params, train_data, val_data, evaluator, dataset_bundle):
        history = wrapper.train(train_data, val_data, **params)
        val_metrics = self._evaluate_classification(wrapper, val_data)
        val_detection = None
        best_threshold = None
        threshold_search = []
        curve_bundle = {}
        threshold_step, distance_threshold = self._metric_protocol_config(dataset_bundle)
        if evaluator is not None and dataset_bundle.val_minerals_df is not None and len(dataset_bundle.val_minerals_df) > 0:
            probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
            best_threshold, val_detection, threshold_search = self._select_best_threshold(
                evaluator,
                probability_map,
                dataset_bundle.val_minerals_df,
            )
            if (
                dataset_bundle.test_minerals_df is not None
                and len(dataset_bundle.test_minerals_df) > 0
                and getattr(dataset_bundle, "test_area_positions", None) is not None
                and len(dataset_bundle.test_area_positions) > 0
            ):
                curve_bundle = self._compute_test_threshold_curves(
                    evaluator,
                    probability_map,
                    dataset_bundle.test_minerals_df,
                    test_area_positions=dataset_bundle.test_area_positions,
                    threshold_step=threshold_step,
                    distance_threshold=distance_threshold,
                )
                if curve_bundle.get("best_test_threshold") is not None:
                    best_threshold = curve_bundle.get("best_test_threshold")
        elif evaluator is not None and dataset_bundle.test_minerals_df is not None and len(dataset_bundle.test_minerals_df) > 0:
            probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
            if getattr(dataset_bundle, "test_area_positions", None) is not None and len(dataset_bundle.test_area_positions) > 0:
                curve_bundle = self._compute_test_threshold_curves(
                    evaluator,
                    probability_map,
                    dataset_bundle.test_minerals_df,
                    test_area_positions=dataset_bundle.test_area_positions,
                    threshold_step=threshold_step,
                    distance_threshold=distance_threshold,
                )
                if curve_bundle.get("best_test_threshold") is not None:
                    best_threshold = curve_bundle.get("best_test_threshold")

        val_detection_rate = get_detection_rate(val_detection)
        val_precision = self._metric_or_default(val_metrics.get("val_precision"), 0.0)
        val_recall = self._metric_or_default(val_metrics.get("val_recall"), 0.0)
        val_sr = self._metric_or_default(val_metrics.get("val_sr"), 0.0)
        val_paf = self._metric_or_default(val_metrics.get("val_paf"), 0.0)
        val_ei = self._metric_or_default(val_metrics.get("val_ei"), 0.0)
        scoring_detection = self._first_available_metric(val_detection_rate, val_recall, default=0.0)
        best_test_ei = curve_bundle.get("best_test_ei")
        composite_score = (
            float(best_test_ei)
            if best_test_ei is not None
            else compute_composite_score(
                scoring_detection,
                val_paf,
                None,
            )
        )
        return {
            "score": composite_score,
            "metrics": {
                "val_accuracy": val_metrics.get("val_accuracy", 0.0),
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_metrics.get("val_f1", 0.0),
                "val_sr": val_sr,
                "val_paf": val_paf,
                "val_ei": val_ei,
                "cv_ei_mean": val_ei,
                "cv_ei_std": 0.0,
                "val_mineral_detection_rate": val_detection_rate,
                "best_detection_threshold": best_threshold,
                "threshold_search": threshold_search,
                "val_mineral_detection": val_detection,
                "test_mineral_detection": curve_bundle.get("best_test_detection"),
                "best_test_threshold": curve_bundle.get("best_test_threshold"),
                "best_test_sr": curve_bundle.get("best_test_sr"),
                "best_test_paf": curve_bundle.get("best_test_paf"),
                "best_test_ei": curve_bundle.get("best_test_ei"),
                "test_threshold_curve": list(curve_bundle.get("threshold_curve") or []),
                "src_curve": list(curve_bundle.get("src_curve") or []),
                "pac_curve": list(curve_bundle.get("pac_curve") or []),
                "metric_protocol": METRIC_PROTOCOL,
                "threshold_strategy": THRESHOLD_STRATEGY,
                "paf_scope": PAF_SCOPE,
                "threshold_rule": THRESHOLD_RULE,
                **metric_protocol_fields(
                    threshold_step=threshold_step,
                    distance_threshold=distance_threshold,
                ),
                "composite_formula": "independent_test_ei" if best_test_ei is not None else resolve_composite_formula(scoring_detection, val_paf, None),
                "composite_score": composite_score,
                "train_acc": history.get("train_acc"),
            },
        }

    def _score_trial_spatial_cv(self, wrapper, params, dataset_bundle):
        spatial_cv = dict(getattr(dataset_bundle, "spatial_cv_splits", {}) or {})
        folds = spatial_cv.get("folds") or []
        dev_data = self._get_final_fit_view(dataset_bundle)
        if dev_data is None:
            raise ValueError("Spatial CV requires development data.")
        dev_x, dev_y = dev_data
        dev_x = np.asarray(dev_x)
        dev_y = np.asarray(dev_y).reshape(-1)

        fold_accuracies = []
        fold_precision_scores = []
        fold_recall_scores = []
        fold_f1_scores = []
        fold_sr_scores = []
        fold_paf_scores = []
        fold_ei_scores = []
        fold_detection_metrics = []
        dev_minerals_df = self._development_minerals_frame(dataset_bundle)
        evaluator = None
        if getattr(dataset_bundle, "h5_path", None) and len(dev_minerals_df) > 0:
            try:
                evaluator = MineralEvaluator(
                    dataset_bundle.h5_path,
                    buffer_radius=dataset_bundle.dataset_summary.get("buffer_radius", 500),
                    prediction_patch_size=dataset_bundle.dataset_summary.get("patch_size"),
                    prediction_patch_stride=dataset_bundle.dataset_summary.get("patch_stride"),
                )
            except Exception as exc:
                self.log_message.emit(f"Spatial CV internal mineral evaluator skipped: {exc}")
        augmentation_config = self._resolve_augmentation_config(dataset_bundle, wrapper)
        for fold in folds:
            train_indices = np.asarray(fold.get("train_indices") or [], dtype=np.int64)
            val_indices = np.asarray(fold.get("val_indices") or [], dtype=np.int64)
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            fold_detection_rate = None
            fold_minerals = []
            fold_train = prepare_training_view(
                wrapper,
                (dev_x[train_indices], dev_y[train_indices]),
                supervised_train_ratio=1.0,
                augmentation_enabled=augmentation_config["augmentation_enabled"],
                augmentation_noise_std=augmentation_config["augmentation_noise_std"],
            )
            fold_val = (dev_x[val_indices], dev_y[val_indices])
            if fold_train is None:
                continue
            wrapper.train(fold_train, fold_val, **params)
            fold_metrics = self._evaluate_classification(wrapper, fold_val)
            fold_ei = float(fold_metrics.get("val_ei", 0.0) or 0.0)
            weight = max(len(val_indices), 1)
            fold_index = int(fold.get("fold_index", len(fold_ei_scores)) or len(fold_ei_scores))
            fold_minerals = self._spatial_cv_fold_minerals(dev_minerals_df, spatial_cv, fold_index)
            if evaluator is not None and len(fold_minerals) > 0:
                try:
                    probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
                    fold_detection = evaluator.evaluate_probability_map(probability_map, fold_minerals)
                    fold_detection_rate = get_detection_rate(fold_detection)
                except Exception as exc:
                    self.log_message.emit(f"Spatial CV fold {fold_index + 1} mineral detection skipped: {exc}")
            fold_accuracies.append((float(fold_metrics.get("val_accuracy", 0.0) or 0.0), weight))
            fold_precision_scores.append((float(fold_metrics.get("val_precision", 0.0) or 0.0), weight))
            fold_recall_scores.append((float(fold_metrics.get("val_recall", 0.0) or 0.0), weight))
            fold_f1_scores.append((float(fold_metrics.get("val_f1", 0.0) or 0.0), weight))
            fold_sr_scores.append((float(fold_metrics.get("val_sr", 0.0) or 0.0), weight))
            fold_paf_scores.append((float(fold_metrics.get("val_paf", 0.0) or 0.0), weight))
            fold_ei_scores.append((fold_ei, weight))
            fold_detection_metrics.append(
                {
                    "fold_index": fold_index,
                    "val_count": int(len(val_indices)),
                    "val_mineral_count": int(len(fold_minerals)),
                    "val_accuracy": float(fold_metrics.get("val_accuracy", 0.0) or 0.0),
                    "val_precision": float(fold_metrics.get("val_precision", 0.0) or 0.0),
                    "val_recall": float(fold_metrics.get("val_recall", 0.0) or 0.0),
                    "val_f1": float(fold_metrics.get("val_f1", 0.0) or 0.0),
                    "val_sr": float(fold_metrics.get("val_sr", 0.0) or 0.0),
                    "val_paf": float(fold_metrics.get("val_paf", 0.0) or 0.0),
                    "val_ei": float(fold_metrics.get("val_ei", 0.0) or 0.0),
                    "val_mineral_detection_rate": fold_detection_rate,
                }
            )

        if not fold_ei_scores:
            raise ValueError("Spatial CV failed to produce valid folds.")

        total_weight = float(sum(weight for _, weight in fold_ei_scores))
        mean_accuracy = sum(value * weight for value, weight in fold_accuracies) / total_weight
        mean_precision = sum(value * weight for value, weight in fold_precision_scores) / total_weight
        mean_recall = sum(value * weight for value, weight in fold_recall_scores) / total_weight
        mean_f1 = sum(value * weight for value, weight in fold_f1_scores) / total_weight
        mean_sr = sum(value * weight for value, weight in fold_sr_scores) / total_weight if fold_sr_scores else 0.0
        mean_paf = sum(value * weight for value, weight in fold_paf_scores) / total_weight if fold_paf_scores else 0.0
        mean_ei = sum(value * weight for value, weight in fold_ei_scores) / total_weight if fold_ei_scores else 0.0
        ei_values = [float(score) for score, _ in fold_ei_scores]
        detection_items = [
            item for item in fold_detection_metrics
            if item.get("val_mineral_detection_rate") is not None
        ]
        if detection_items:
            detection_values = np.asarray(
                [float(item.get("val_mineral_detection_rate", 0.0) or 0.0) for item in detection_items],
                dtype=np.float64,
            )
            detection_weights = np.asarray(
                [max(int(item.get("val_mineral_count", 0)), 1) for item in detection_items],
                dtype=np.float64,
            )
            mean_val_detection_rate = float(np.average(detection_values, weights=detection_weights))
        else:
            mean_val_detection_rate = None

        return {
            "score": float(mean_ei),
            "metrics": {
                "val_accuracy": float(mean_accuracy),
                "val_precision": float(mean_precision),
                "val_recall": float(mean_recall),
                "val_f1": float(mean_f1),
                "val_sr": float(mean_sr),
                "val_paf": float(mean_paf),
                "val_ei": float(mean_ei),
                "cv_ei_mean": float(mean_ei),
                "cv_ei_std": float(np.std(ei_values, ddof=0)) if ei_values else 0.0,
                "val_mineral_detection_rate": mean_val_detection_rate,
                "best_detection_threshold": None,
                "threshold_search": [],
                "val_mineral_detection": None,
                "composite_formula": "cv_ei_mean",
                "composite_score": float(mean_ei),
                "cv_fold_scores": ei_values,
                "cv_fold_ei_scores": ei_values,
                "cv_fold_metrics": fold_detection_metrics,
                "cv_fold_count": int(len(fold_ei_scores)),
                "cv_axis": spatial_cv.get("axis_name"),
                "cv_buffer_distance": float(spatial_cv.get("buffer_distance", 0.0) or 0.0),
            },
        }

    def _select_split_views(self, wrapper, dataset_bundle):
        augmentation_config = self._resolve_augmentation_config(dataset_bundle, wrapper)
        return prepare_model_views(
            wrapper,
            dataset_bundle.train_loader if dataset_bundle.train_loader is not None else dataset_bundle.train_data_array,
            dataset_bundle.val_loader if dataset_bundle.val_loader is not None else dataset_bundle.val_data_array,
            dataset_bundle.test_loader if dataset_bundle.test_loader is not None else dataset_bundle.test_data_array,
            supervised_train_ratio=1.0,
            augmentation_enabled=augmentation_config["augmentation_enabled"],
            augmentation_noise_std=augmentation_config["augmentation_noise_std"],
        )

    def _extract_labels(self, data):
        if data is None:
            return np.array([], dtype=np.int64)
        if isinstance(data, tuple):
            return (np.asarray(data[1]).reshape(-1) > 0).astype(np.int64)

        labels = []
        for _, batch_labels in data:
            labels.append((batch_labels.cpu().numpy().reshape(-1) > 0).astype(np.int64))
        if not labels:
            return np.array([], dtype=np.int64)
        return np.concatenate(labels, axis=0)

    def _evaluate_classification(self, wrapper, data, prefix="val"):
        if data is None:
            return {
                f"{prefix}_accuracy": None,
                f"{prefix}_precision": None,
                f"{prefix}_recall": None,
                f"{prefix}_f1": None,
            }

        labels = self._extract_labels(data)
        if len(labels) == 0:
            return {
                f"{prefix}_accuracy": None,
                f"{prefix}_precision": None,
                f"{prefix}_recall": None,
                f"{prefix}_f1": None,
            }

        predictions = (np.asarray(wrapper.predict(data)).reshape(-1) > 0).astype(np.int64)
        accuracy = float(np.mean(predictions == labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        ei_metrics = compute_sr_paf_ei(labels, predictions)
        return {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_precision": float(precision),
            f"{prefix}_recall": float(recall),
            f"{prefix}_f1": float(f1),
            f"{prefix}_sr": float(ei_metrics.get("sr", 0.0)),
            f"{prefix}_paf": float(ei_metrics.get("paf", 0.0)),
            f"{prefix}_ei": float(ei_metrics.get("ei", 0.0)),
        }

    def _best_result_by(self, metric_key: str, *, fallback_key: str | None = None):
        candidates = [item for item in self.results if item is not None]
        if not candidates:
            return None

        def _metric(item):
            value = item.get(metric_key)
            if value in (None, "") and fallback_key:
                value = item.get(fallback_key)
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return max(candidates, key=_metric)

    def stop(self):
        self.is_running = False
        if self.current_wrapper is not None and hasattr(self.current_wrapper, "stop"):
            self.current_wrapper.stop()

    def get_best_model(self):
        if not self.results:
            return None
        return self.results[0]

    def export_results(self, output_path):
        if not self.results:
            return

        df = pd.DataFrame(
            [
                {
                    "模型": item["model_name"],
                    "测试集EI": f"{primary_test_ei_score(item):.4f}",
                    "综合评分": f"{item.get('composite_score', 0.0):.4f}",
                    "验证准确率": f"{item.get('val_accuracy', 0.0):.4f}",
                    "验证F1": f"{item.get('val_f1', 0.0):.4f}",
                    "内部验证矿点检出率": "" if item.get("val_mineral_detection_rate") is None else f"{item['val_mineral_detection_rate']:.4f}",
                    "测试准确率": "" if item.get("test_accuracy") is None else f"{item['test_accuracy']:.4f}",
                    "测试F1": "" if item.get("test_f1") is None else f"{item['test_f1']:.4f}",
                    "测试矿点检出率": "" if item.get("test_mineral_detection_rate") is None else f"{item['test_mineral_detection_rate']:.4f}",
                    "测试最优阈值": "" if item.get("best_test_threshold") is None else f"{float(item.get('best_test_threshold')):.4f}",
                    "测试最优阈值EI": "" if item.get("best_test_ei") is None else f"{float(item.get('best_test_ei')):.4f}",
                    "SRC点数": int(len(item.get("src_curve") or [])),
                    "PAC点数": int(len(item.get("pac_curve") or [])),
                    "最优Trial分数": f"{item.get('best_score', 0.0):.4f}",
                    "训练时间(秒)": f"{item.get('training_time', 0.0):.1f}",
                    "最优参数": str(item.get("best_params", {})),
                    "改进建议": "；".join(item.get("improvement_advice", [])),
                }
                for item in self.results
            ]
        )
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
