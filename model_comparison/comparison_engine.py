"""Core execution engine for model comparison experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:  # pragma: no cover - 运行环境可能未安装 torch
    class _TorchFallback:
        class cuda:
            @staticmethod
            def is_available():
                return False

    torch = _TorchFallback()
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.metrics import precision_recall_fscore_support

from .analysis_utils import (
    build_improvement_advice,
    compute_composite_score,
    compute_sr_paf_ei,
    get_detection_rate,
    primary_test_ei_score,
    primary_test_ei_sort_key,
    resolve_composite_formula,
    to_jsonable,
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
from .model_wrappers import BaseModelWrapper


class ComparisonEngine(QObject):
    """Manage experiment execution and aggregate comparison results."""

    experiment_started = pyqtSignal(str, int)
    task_started = pyqtSignal(int, str)
    task_progress = pyqtSignal(int, int, int)
    task_completed = pyqtSignal(int, dict)
    experiment_completed = pyqtSignal(list)
    log_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.experiments = []
        self.results = []
        self.is_running = False
        self.should_stop = False
        self.current_trainer = None
        self.current_dataset_summary = {}
        self.h5_path = None
        self.train_minerals_df = None
        self.val_minerals_df = None
        self.test_minerals_df = None
        self.dev_data_array = None
        self.spatial_cv_splits = None
        self.test_area_positions = None

    def add_experiment(self, model_type, model_class, config, experiment_name=None):
        if experiment_name is None:
            experiment_name = f"{model_type}_{len(self.experiments)}"

        experiment = {
            "id": len(self.experiments),
            "name": experiment_name,
            "model_type": model_type,
            "model_object": model_class,
            "config": config.copy(),
            "status": "pending",
        }
        self.experiments.append(experiment)
        self.log_message.emit(f"添加实验: {experiment_name}")

    def clear_experiments(self):
        self.experiments = []
        self.results = []

    @staticmethod
    def _build_model_context(wrapper, fallback_name: str = "") -> dict:
        return {
            "model_name": getattr(wrapper, "model_name", fallback_name),
            "model_family": getattr(wrapper, "model_family", "unknown"),
            "input_kind": getattr(wrapper, "input_kind", "vector"),
            "label_mode": getattr(wrapper, "label_mode", "binary"),
            "training_mode": getattr(wrapper, "training_mode", "supervised"),
            "is_pu_model": bool(getattr(wrapper, "is_pu_model", False)),
        }

    @staticmethod
    def _to_scalar_metric(value, default: float = 0.0) -> float:
        """Convert metric containers (list/tuple/ndarray/scalar) into one float."""
        if value is None:
            return float(default)
        if isinstance(value, (list, tuple)):
            if not value:
                return float(default)
            return ComparisonEngine._to_scalar_metric(value[-1], default=default)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float(default)
            flat = value.reshape(-1)
            return ComparisonEngine._to_scalar_metric(flat[-1], default=default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _threshold_candidates(self):
        return [float(value) for value in threshold_candidates(step=DEFAULT_THRESHOLD_STEP)]

    def _metric_protocol_config(self):
        summary = dict(self.current_dataset_summary or {})
        protocol = dict(summary.get("evaluation_protocol") or {})
        return (
            float(protocol.get("threshold_step", DEFAULT_THRESHOLD_STEP) or DEFAULT_THRESHOLD_STEP),
            float(protocol.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD) or DEFAULT_DISTANCE_THRESHOLD),
        )

    def _compute_test_threshold_curves(self, evaluator, probability_map, test_minerals_df):
        if evaluator is None or probability_map is None or test_minerals_df is None or len(test_minerals_df) == 0:
            return {}
        if self.test_area_positions is None or len(self.test_area_positions) == 0:
            return {}
        threshold_step, distance_threshold = self._metric_protocol_config()
        try:
            metrics = evaluate_independent_test_metrics(
                probability_map,
                getattr(evaluator, "metadata", {}) or {},
                test_minerals_df,
                test_positions=self.test_area_positions,
                threshold_step=threshold_step,
                distance_threshold=distance_threshold,
            )
        except Exception as exc:
            self.log_message.emit(f"独立测试集指标计算跳过: {exc}")
            return {}
        best_row = dict(metrics.get("best") or {})
        threshold_curve = list(metrics.get("threshold_curve") or [])
        best_detection = detection_from_metric_row(best_row)
        return {
            "best_test_threshold": None if best_row is None else float(best_row["threshold"]),
            "best_test_detection": best_detection,
            "best_test_sr": None if best_row is None else float(best_row["test_sr"]),
            "best_test_paf": None if best_row is None else float(best_row["test_paf"]),
            "best_test_ei": None if best_row is None else float(best_row["test_ei"]),
            "test_threshold_curve": threshold_curve,
            "src_curve": list(metrics.get("src_curve") or []),
            "pac_curve": list(metrics.get("pac_curve") or []),
            "metric_protocol": METRIC_PROTOCOL,
            "threshold_strategy": THRESHOLD_STRATEGY,
            "paf_scope": PAF_SCOPE,
            "threshold_rule": THRESHOLD_RULE,
            **metric_protocol_fields(
                threshold_step=threshold_step,
                distance_threshold=distance_threshold,
            ),
        }

    def stop(self):
        self.should_stop = True
        self.log_message.emit("正在停止实验...")
        if self.current_trainer is not None and hasattr(self.current_trainer, "stop"):
            try:
                self.current_trainer.stop()
            except Exception as exc:
                self.log_message.emit(f"停止训练器时出现警告: {exc}")

    def run_experiments(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        dev_data_array=None,
        spatial_cv_splits=None,
        dataset_summary=None,
        h5_path=None,
        train_minerals_df=None,
        val_minerals_df=None,
        test_minerals_df=None,
        test_area_positions=None,
        parallel=False,
    ):
        del parallel

        if self.is_running:
            self.log_message.emit("实验已在运行中。")
            return

        if not self.experiments:
            self.log_message.emit("没有待运行的实验。")
            return

        self.is_running = True
        self.should_stop = False
        self.results = []
        self.current_dataset_summary = dict(dataset_summary or {})
        self.h5_path = h5_path
        self.train_minerals_df = train_minerals_df
        self.val_minerals_df = val_minerals_df
        self.test_minerals_df = test_minerals_df
        self.test_area_positions = None if test_area_positions is None else np.asarray(test_area_positions, dtype=np.float64)
        self.dev_data_array = dev_data_array
        self.spatial_cv_splits = dict(spatial_cv_splits or {})

        total_experiments = len(self.experiments)
        self.experiment_started.emit("模型对比实验", total_experiments)

        try:
            for experiment in self.experiments:
                if self.should_stop:
                    break

                experiment_id = experiment["id"]
                self.task_started.emit(experiment_id, experiment["name"])
                self.log_message.emit(
                    f"\n开始实验 {experiment_id + 1}/{total_experiments}: {experiment['name']}"
                )

                try:
                    result = self._run_single_experiment(
                        experiment=experiment,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        experiment_id=experiment_id,
                    )
                    experiment["status"] = "completed"
                    self.results.append(result)
                    self.task_completed.emit(experiment_id, result)
                    self.log_message.emit(f"实验 {experiment['name']} 完成")
                except Exception as exc:
                    experiment["status"] = "failed"
                    error_message = f"实验 {experiment['name']} 失败: {exc}"
                    self.log_message.emit(error_message)
                    self.error_occurred.emit(error_message)
        finally:
            self.current_trainer = None
            self.is_running = False
            self.dev_data_array = None
            self.spatial_cv_splits = None
            self.train_minerals_df = None
            self.test_area_positions = None
            self.results.sort(key=primary_test_ei_sort_key)
            self.experiment_completed.emit(self.results)
            self.log_message.emit(f"\n全部实验结束，成功 {len(self.results)} 个。")

    def _run_single_experiment(self, experiment, train_loader, val_loader, test_loader, experiment_id):
        model_object = experiment.get("model_object", experiment.get("model_class"))
        config = experiment["config"]

        if isinstance(model_object, BaseModelWrapper) or (
            hasattr(model_object, "train")
            and hasattr(model_object, "predict")
            and hasattr(model_object, "predict_proba")
            and hasattr(model_object, "data_mode")
        ):
            wrapper = model_object
            if self._has_spatial_cv():
                return self._run_single_experiment_spatial_cv(
                    experiment=experiment,
                    wrapper=wrapper,
                    config=config,
                    test_loader=test_loader,
                    experiment_id=experiment_id,
                )
            train_data, val_data, test_data = self._select_wrapper_views(
                wrapper,
                train_loader,
                val_loader,
                test_loader,
                config=config,
            )
            start_time = time.time()
            self.current_trainer = wrapper
            try:
                history = wrapper.train(train_data, val_data, **config)
            finally:
                training_time = time.time() - start_time
                self.current_trainer = None

            train_acc = self._to_scalar_metric((history or {}).get("train_acc", 0.0), default=0.0)
            val_acc = self._to_scalar_metric((history or {}).get("val_acc", 0.0), default=0.0)
            val_loss = self._to_min_metric((history or {}).get("val_loss", 0.0), default=0.0)
            val_metrics = self._evaluate_on_test(wrapper, val_data, None) if val_data is not None else {}

            result = {
                "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_id}",
                "model_type": experiment["model_type"],
                "model_name": getattr(wrapper, "model_name", experiment["model_type"]),
                "_trained_wrapper": wrapper,
                "model_context": self._build_model_context(wrapper, experiment["model_type"]),
                "dataset_meta": {
                    "input_channels": config.get("input_channels"),
                    "image_size": config.get("image_size"),
                    "num_classes": config.get("num_classes", 2),
                    "input_kind": getattr(wrapper, "input_kind", "vector"),
                    "label_mode": getattr(wrapper, "label_mode", "binary"),
                },
                "config": config,
                "training_config": dict(getattr(wrapper, "training_config", {}) or config),
                "dataset_summary": dict(self.current_dataset_summary or {}),
                "results": {
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_recall": float(val_metrics.get("test_recall", 0.0) or 0.0),
                    "val_paf": float(val_metrics.get("test_paf", 0.0) or 0.0),
                    "val_ei": float(val_metrics.get("test_ei", 0.0) or 0.0),
                    "training_time_seconds": training_time,
                },
                "test_metrics_available": False,
                "mineral_eval_available": False,
                "val_mineral_detection": None,
                "test_mineral_detection": None,
            }

            if test_data is not None:
                test_metrics = self._evaluate_on_test(wrapper, test_data, None)
            result["results"].update(test_metrics)
            result["results"]["generalization_score"] = test_metrics["test_acc"] / max(train_acc, 0.01)
            result["test_metrics_available"] = True

            if self.h5_path and (
                (self.val_minerals_df is not None and len(self.val_minerals_df) > 0)
                or (self.test_minerals_df is not None and len(self.test_minerals_df) > 0)
            ):
                try:
                    evaluator = MineralEvaluator(
                        self.h5_path,
                        buffer_radius=self.current_dataset_summary.get("buffer_radius", 500),
                        prediction_patch_size=config.get(
                            "prediction_patch_size",
                            self.current_dataset_summary.get("patch_size"),
                        ),
                        prediction_patch_stride=config.get(
                            "prediction_patch_stride",
                            self.current_dataset_summary.get("patch_stride"),
                        ),
                        prediction_batch_size=config.get("prediction_batch_size", 128),
                    )
                    probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
                    result["val_mineral_detection"] = evaluator.evaluate_probability_map(
                        probability_map,
                        self.val_minerals_df,
                    )
                    curve_bundle = self._compute_test_threshold_curves(
                        evaluator,
                        probability_map,
                        self.test_minerals_df,
                    )
                    if curve_bundle:
                        result.update(curve_bundle)
                        result["test_mineral_detection"] = curve_bundle.get("best_test_detection")
                        result["best_detection_threshold"] = curve_bundle.get("best_test_threshold")
                        result["test_sr"] = curve_bundle.get("best_test_sr")
                        result["test_paf"] = curve_bundle.get("best_test_paf")
                        result["test_ei"] = curve_bundle.get("best_test_ei")
                        result["results"]["test_sr"] = curve_bundle.get("best_test_sr")
                        result["results"]["test_paf"] = curve_bundle.get("best_test_paf")
                        result["results"]["test_ei"] = curve_bundle.get("best_test_ei")
                    else:
                        result["test_mineral_detection"] = evaluator.evaluate_probability_map(
                            probability_map,
                            self.test_minerals_df,
                        )
                    result["mineral_eval_available"] = bool(
                        result["val_mineral_detection"] or result["test_mineral_detection"]
                    )
                except Exception as exc:
                    self.log_message.emit(f"矿点检测评估已跳过: {exc}")

            result["val_mineral_detection_rate"] = get_detection_rate(result["val_mineral_detection"])
            result["test_mineral_detection_rate"] = get_detection_rate(result["test_mineral_detection"])
            scoring_detection = float(result["val_mineral_detection_rate"] or 0.0)
            if result.get("best_test_ei") is not None:
                result["composite_formula"] = "independent_test_ei"
                result["composite_score"] = float(result.get("best_test_ei") or 0.0)
            else:
                result["composite_formula"] = resolve_composite_formula(
                    scoring_detection,
                    result["results"].get("val_paf"),
                    None,
                )
                result["composite_score"] = compute_composite_score(
                    scoring_detection,
                    result["results"].get("val_paf"),
                    None,
                )
            result["metric_protocol"] = METRIC_PROTOCOL
            result["threshold_strategy"] = THRESHOLD_STRATEGY
            result["paf_scope"] = PAF_SCOPE
            result["threshold_rule"] = THRESHOLD_RULE
            result["primary_score"] = primary_test_ei_score(result)
            result["selection_strategy"] = "test_ei"
            result["search_boundary_hits"] = []
            result["improvement_advice"] = build_improvement_advice(result)

            return result

        from ..cnn.trainer import Trainer

        model = model_object(
            num_classes=config["num_classes"],
            input_channels=config["input_channels"],
            image_size=config.get("image_size", 64),
        )

        device = config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        start_time = time.time()
        scheduler_name = config.get("scheduler")
        scheduler_enabled = bool(scheduler_name) and str(scheduler_name).lower() not in {"none", "off", "disabled"}
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs"],
            lr=config["learning_rate"],
            device=device,
            image_size=config.get("image_size", 64),
            save_path=config.get("save_path", "./models/comparison"),
            optimizer_name=config.get("optimizer", "AdamW"),
            scheduler_enabled=scheduler_enabled,
            scheduler_type=scheduler_name or "CosineAnnealingLR",
        )

        self.current_trainer = trainer
        trainer.log_message.connect(self.log_message.emit)
        trainer.epoch_completed.connect(
            lambda epoch, *_: self.task_progress.emit(experiment_id, epoch, config["epochs"])
        )

        trainer.train()
        training_time = time.time() - start_time
        model.normalization_stats = getattr(train_loader, "normalization_stats", None)
        model.split_info = getattr(train_loader, "split_info", None)

        train_acc_history = trainer.history.get("train_acc", [])
        val_acc_history = trainer.history.get("val_acc", [])
        val_loss_history = trainer.history.get("val_loss", [])

        result = {
            "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_id}",
            "model_type": experiment["model_type"],
            "model_name": getattr(model, "display_name", experiment["model_type"]),
            "_trained_model": model,
            "model_context": self._build_model_context(model, experiment["model_type"]),
            "dataset_meta": {
                "input_channels": config.get("input_channels"),
                "image_size": config.get("image_size"),
                "num_classes": config.get("num_classes", 2),
                "input_kind": getattr(model, "input_kind", "vector"),
                "label_mode": getattr(model, "label_mode", "binary"),
            },
            "config": config,
            "training_config": dict(getattr(model, "training_config", {}) or config),
            "dataset_summary": dict(self.current_dataset_summary or {}),
            "results": {
                "train_acc": max(train_acc_history) if train_acc_history else 0.0,
                "val_acc": max(val_acc_history) if val_acc_history else 0.0,
                "val_loss": min(val_loss_history) if val_loss_history else 0.0,
                "training_time_seconds": training_time,
            },
            "test_metrics_available": False,
            "mineral_eval_available": False,
            "val_mineral_detection": None,
            "test_mineral_detection": None,
        }

        if test_loader is not None:
            test_metrics = self._evaluate_on_test(model, test_loader, device)
            result["results"].update(test_metrics)
            result["results"]["generalization_score"] = test_metrics["test_acc"] / max(
                result["results"]["train_acc"],
                0.01,
            )
            result["test_metrics_available"] = True

        if self.h5_path and (
            (self.val_minerals_df is not None and len(self.val_minerals_df) > 0)
            or (self.test_minerals_df is not None and len(self.test_minerals_df) > 0)
        ):
            try:
                evaluator = MineralEvaluator(
                    self.h5_path,
                    buffer_radius=self.current_dataset_summary.get("buffer_radius", 500),
                    prediction_patch_size=config.get(
                        "prediction_patch_size",
                        self.current_dataset_summary.get("patch_size"),
                    ),
                    prediction_patch_stride=config.get(
                        "prediction_patch_stride",
                        self.current_dataset_summary.get("patch_stride"),
                    ),
                    prediction_batch_size=config.get("prediction_batch_size", 128),
                )
                probability_map = evaluator.generate_probability_map_for_torch_model(
                    model,
                    device,
                    config,
                )
                result["val_mineral_detection"] = evaluator.evaluate_probability_map(
                    probability_map,
                    self.val_minerals_df,
                )
                curve_bundle = self._compute_test_threshold_curves(
                    evaluator,
                    probability_map,
                    self.test_minerals_df,
                )
                if curve_bundle:
                    result.update(curve_bundle)
                    result["test_mineral_detection"] = curve_bundle.get("best_test_detection")
                    result["best_detection_threshold"] = curve_bundle.get("best_test_threshold")
                    result["test_sr"] = curve_bundle.get("best_test_sr")
                    result["test_paf"] = curve_bundle.get("best_test_paf")
                    result["test_ei"] = curve_bundle.get("best_test_ei")
                    result["results"]["test_sr"] = curve_bundle.get("best_test_sr")
                    result["results"]["test_paf"] = curve_bundle.get("best_test_paf")
                    result["results"]["test_ei"] = curve_bundle.get("best_test_ei")
                else:
                    result["test_mineral_detection"] = evaluator.evaluate_probability_map(
                        probability_map,
                        self.test_minerals_df,
                    )
                result["mineral_eval_available"] = bool(
                    result["val_mineral_detection"] or result["test_mineral_detection"]
                )
            except Exception as exc:
                self.log_message.emit(f"矿点检测评估已跳过: {exc}")

        result["val_mineral_detection_rate"] = get_detection_rate(result["val_mineral_detection"])
        result["test_mineral_detection_rate"] = get_detection_rate(result["test_mineral_detection"])
        result["val_recall"] = float(result["results"].get("val_recall", result["val_mineral_detection_rate"] or 0.0) or 0.0)
        result["test_recall"] = float(result["results"].get("test_recall", result["test_mineral_detection_rate"] or 0.0) or 0.0)
        scoring_detection = float(result["val_mineral_detection_rate"] or result["val_recall"] or 0.0)
        if result.get("best_test_ei") is not None:
            result["composite_formula"] = "independent_test_ei"
            result["composite_score"] = float(result.get("best_test_ei") or 0.0)
        else:
            result["composite_formula"] = resolve_composite_formula(
                scoring_detection,
                result["results"].get("val_paf"),
                None,
            )
            result["composite_score"] = compute_composite_score(
                scoring_detection,
                result["results"].get("val_paf"),
                None,
            )
        result["metric_protocol"] = METRIC_PROTOCOL
        result["threshold_strategy"] = THRESHOLD_STRATEGY
        result["paf_scope"] = PAF_SCOPE
        result["threshold_rule"] = THRESHOLD_RULE
        result["primary_score"] = primary_test_ei_score(result)
        result["selection_strategy"] = "test_ei"
        result["search_boundary_hits"] = []
        result["improvement_advice"] = build_improvement_advice(result)

        self.current_trainer = None
        return result

    def _has_spatial_cv(self) -> bool:
        spatial_cv = dict(self.spatial_cv_splits or {})
        folds = spatial_cv.get("folds") or []
        return bool(self.dev_data_array is not None and folds)

    @staticmethod
    def _to_min_metric(value, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if isinstance(value, (list, tuple, np.ndarray)):
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            array = array[np.isfinite(array)]
            if array.size == 0:
                return float(default)
            return float(np.min(array))
        try:
            metric = float(value)
        except (TypeError, ValueError):
            return float(default)
        return metric if np.isfinite(metric) else float(default)

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

    def _development_minerals_frame(self):
        frames = []
        for frame in (self.train_minerals_df, self.val_minerals_df):
            if frame is not None and len(frame) > 0:
                frames.append(frame.reset_index(drop=True))
        if not frames:
            return pd.DataFrame(columns=["x", "y"])
        return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    def _spatial_cv_fold_minerals(self, minerals_df, fold_index):
        if minerals_df is None or len(minerals_df) == 0:
            return pd.DataFrame(columns=[] if minerals_df is None else minerals_df.columns)
        spatial_cv = dict(self.spatial_cv_splits or {})
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

    @staticmethod
    def _to_array_pair(data):
        if data is None:
            return None
        if isinstance(data, tuple):
            if len(data) != 2:
                raise TypeError("Tuple inputs must be (features, labels).")
            return np.asarray(data[0]), np.asarray(data[1]).reshape(-1)
        features = []
        labels = []
        for batch_inputs, batch_labels in data:
            features.append(np.asarray(batch_inputs.detach().cpu().numpy()))
            labels.append(np.asarray(batch_labels.detach().cpu().numpy()).reshape(-1))
        if not features:
            return None
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0).reshape(-1)

    def _evaluate_binary_metrics(self, wrapper, data, prefix: str):
        metrics = self._evaluate_on_test(wrapper, data, None)
        return {
            f"{prefix}_accuracy": float(metrics.get("test_acc", 0.0) or 0.0),
            f"{prefix}_recall": float(metrics.get("test_recall", 0.0) or 0.0),
            f"{prefix}_f1": float(metrics.get("test_f1", 0.0) or 0.0),
            f"{prefix}_sr": float(metrics.get("test_sr", 0.0) or 0.0),
            f"{prefix}_paf": float(metrics.get("test_paf", 0.0) or 0.0),
            f"{prefix}_ei": float(metrics.get("test_ei", 0.0) or 0.0),
        }

    def _run_single_experiment_spatial_cv(self, experiment, wrapper, config, test_loader, experiment_id):
        spatial_cv = dict(self.spatial_cv_splits or {})
        folds = list(spatial_cv.get("folds") or [])
        dev_data = self._to_array_pair(self.dev_data_array)
        if dev_data is None:
            raise ValueError("Spatial CV requires development data.")
        dev_x, dev_y = dev_data
        dev_x = np.asarray(dev_x)
        dev_y = np.asarray(dev_y).reshape(-1)
        fold_count = int(len(folds))
        if fold_count <= 0:
            raise ValueError("Spatial CV folds are empty.")

        fold_metrics = []
        total_training_time = 0.0
        dev_minerals_df = self._development_minerals_frame()
        evaluator = None
        if self.h5_path and len(dev_minerals_df) > 0:
            try:
                evaluator = MineralEvaluator(
                    self.h5_path,
                    buffer_radius=self.current_dataset_summary.get("buffer_radius", 500),
                    prediction_patch_size=config.get(
                        "prediction_patch_size",
                        self.current_dataset_summary.get("patch_size"),
                    ),
                    prediction_patch_stride=config.get(
                        "prediction_patch_stride",
                        self.current_dataset_summary.get("patch_stride"),
                    ),
                    prediction_batch_size=config.get("prediction_batch_size", 128),
                )
            except Exception as exc:
                self.log_message.emit(f"空间 CV 内部验证矿点评估初始化失败，已跳过: {exc}")
        for fold_order, fold in enumerate(folds, start=1):
            if self.should_stop:
                break
            train_indices = np.asarray(fold.get("train_indices") or [], dtype=np.int64)
            val_indices = np.asarray(fold.get("val_indices") or [], dtype=np.int64)
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            fold_detection_rate = None
            fold_minerals = []

            fold_train_raw = (dev_x[train_indices], dev_y[train_indices])
            fold_val_raw = (dev_x[val_indices], dev_y[val_indices])
            fold_train, fold_val, _ = prepare_model_views(
                wrapper,
                fold_train_raw,
                fold_val_raw,
                None,
                supervised_train_ratio=1.0,
                augmentation_enabled=bool(config.get("augmentation_enabled", False)),
                augmentation_noise_std=float(config.get("augmentation_noise_std", 0.01) or 0.01),
            )

            self.current_trainer = wrapper
            fold_start = time.time()
            try:
                history = wrapper.train(fold_train, fold_val, **config)
            finally:
                total_training_time += time.time() - fold_start
                self.current_trainer = None

            val_scores = self._evaluate_binary_metrics(wrapper, fold_val, prefix="val")
            val_loss = self._to_min_metric((history or {}).get("val_loss", 0.0), default=0.0)
            fold_minerals = self._spatial_cv_fold_minerals(
                dev_minerals_df,
                int(fold.get("fold_index", fold_order - 1) or (fold_order - 1)),
            )
            if evaluator is not None and len(fold_minerals) > 0:
                try:
                    probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
                    fold_detection = evaluator.evaluate_probability_map(probability_map, fold_minerals)
                    fold_detection_rate = get_detection_rate(fold_detection)
                except Exception as exc:
                    self.log_message.emit(f"第 {fold_order} 折内部验证矿点检出率评估失败，已跳过: {exc}")
            fold_metrics.append(
                {
                    "fold_index": int(fold.get("fold_index", fold_order - 1) or (fold_order - 1)),
                    "val_count": int(len(val_indices)),
                    "val_mineral_count": int(len(fold_minerals)),
                    "train_acc": self._to_scalar_metric((history or {}).get("train_acc", 0.0), default=0.0),
                    "val_accuracy": float(val_scores.get("val_accuracy", 0.0) or 0.0),
                    "val_recall": float(val_scores.get("val_recall", 0.0) or 0.0),
                    "val_loss": float(val_loss),
                    "val_f1": float(val_scores.get("val_f1", 0.0) or 0.0),
                    "val_sr": float(val_scores.get("val_sr", 0.0) or 0.0),
                    "val_paf": float(val_scores.get("val_paf", 0.0) or 0.0),
                    "val_ei": float(val_scores.get("val_ei", 0.0) or 0.0),
                    "val_mineral_detection_rate": fold_detection_rate,
                }
            )
            self.task_progress.emit(experiment_id, fold_order, fold_count)

        if not fold_metrics:
            raise ValueError("Spatial CV failed to produce valid folds.")

        weights = np.asarray([max(int(item.get("val_count", 0)), 1) for item in fold_metrics], dtype=np.float64)

        def _weighted_mean(key: str) -> float:
            values = np.asarray([float(item.get(key, 0.0) or 0.0) for item in fold_metrics], dtype=np.float64)
            return float(np.average(values, weights=weights))

        def _fold_values(key: str) -> np.ndarray:
            values = np.asarray([float(item.get(key, 0.0) or 0.0) for item in fold_metrics], dtype=np.float64)
            return values[np.isfinite(values)]

        def _fold_mean(key: str) -> float:
            values = _fold_values(key)
            return float(np.mean(values)) if len(values) > 0 else 0.0

        def _fold_std(key: str) -> float:
            values = _fold_values(key)
            return float(np.std(values, ddof=0)) if len(values) > 0 else 0.0

        mean_train_acc = _weighted_mean("train_acc")
        mean_val_acc = _weighted_mean("val_accuracy")
        mean_val_recall = _weighted_mean("val_recall")
        mean_val_loss = _weighted_mean("val_loss")
        mean_val_f1 = _weighted_mean("val_f1")
        mean_val_sr = _weighted_mean("val_sr")
        mean_val_paf = _weighted_mean("val_paf")
        mean_val_ei = _weighted_mean("val_ei")
        detection_items = [
            item for item in fold_metrics
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
            mean_val_mineral_detection_rate = float(np.average(detection_values, weights=detection_weights))
        else:
            mean_val_mineral_detection_rate = None
        cv_sr_mean = _fold_mean("val_sr")
        cv_sr_std = _fold_std("val_sr")
        cv_paf_mean = _fold_mean("val_paf")
        cv_paf_std = _fold_std("val_paf")
        cv_ei_mean = _fold_mean("val_ei")
        cv_ei_std = _fold_std("val_ei")

        final_train = prepare_training_view(
            wrapper,
            (dev_x, dev_y),
            supervised_train_ratio=1.0,
            augmentation_enabled=bool(config.get("augmentation_enabled", False)),
            augmentation_noise_std=float(config.get("augmentation_noise_std", 0.01) or 0.01),
        )
        self.current_trainer = wrapper
        final_fit_start = time.time()
        try:
            wrapper.train(final_train, None, **config)
        finally:
            total_training_time += time.time() - final_fit_start
            self.current_trainer = None

        test_data = test_loader if test_loader is not None else None
        test_metrics = self._evaluate_on_test(wrapper, test_data, None) if test_data is not None else {}

        result = {
            "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_id}",
            "model_type": experiment["model_type"],
            "model_name": getattr(wrapper, "model_name", experiment["model_type"]),
            "_trained_wrapper": wrapper,
            "model_context": self._build_model_context(wrapper, experiment["model_type"]),
            "dataset_meta": {
                "input_channels": config.get("input_channels"),
                "image_size": config.get("image_size"),
                "num_classes": config.get("num_classes", 2),
                "input_kind": getattr(wrapper, "input_kind", "vector"),
                "label_mode": getattr(wrapper, "label_mode", "binary"),
            },
            "config": config,
            "training_config": dict(getattr(wrapper, "training_config", {}) or config),
            "dataset_summary": dict(self.current_dataset_summary or {}),
            "results": {
                "train_acc": float(mean_train_acc),
                "val_acc": float(mean_val_acc),
                "val_recall": float(mean_val_recall),
                "val_loss": float(mean_val_loss),
                "val_f1": float(mean_val_f1),
                "val_sr": float(mean_val_sr),
                "val_paf": float(mean_val_paf),
                "val_ei": float(mean_val_ei),
                "cv_sr_mean": float(cv_sr_mean),
                "cv_sr_std": float(cv_sr_std),
                "cv_paf_mean": float(cv_paf_mean),
                "cv_paf_std": float(cv_paf_std),
                "cv_ei_mean": float(cv_ei_mean),
                "cv_ei_std": float(cv_ei_std),
                "cv_fold_count": int(len(fold_metrics)),
                "training_time_seconds": float(total_training_time),
            },
            "test_metrics_available": bool(test_data is not None),
            "mineral_eval_available": False,
            "val_mineral_detection": None,
            "test_mineral_detection": None,
            "cv_fold_metrics": fold_metrics,
            "cv_axis": spatial_cv.get("axis_name"),
            "cv_buffer_distance": float(spatial_cv.get("buffer_distance", 0.0) or 0.0),
        }

        if test_data is not None:
            result["results"].update(test_metrics)
            result["results"]["generalization_score"] = float(test_metrics.get("test_acc", 0.0) or 0.0) / max(
                result["results"]["train_acc"],
                0.01,
            )

        if self.h5_path and self.test_minerals_df is not None and len(self.test_minerals_df) > 0:
            try:
                evaluator = MineralEvaluator(
                    self.h5_path,
                    buffer_radius=self.current_dataset_summary.get("buffer_radius", 500),
                    prediction_patch_size=config.get(
                        "prediction_patch_size",
                        self.current_dataset_summary.get("patch_size"),
                    ),
                    prediction_patch_stride=config.get(
                        "prediction_patch_stride",
                        self.current_dataset_summary.get("patch_stride"),
                    ),
                    prediction_batch_size=config.get("prediction_batch_size", 128),
                )
                probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
                curve_bundle = self._compute_test_threshold_curves(
                    evaluator,
                    probability_map,
                    self.test_minerals_df,
                )
                if curve_bundle:
                    result.update(curve_bundle)
                    result["test_mineral_detection"] = curve_bundle.get("best_test_detection")
                    result["best_detection_threshold"] = curve_bundle.get("best_test_threshold")
                    result["test_sr"] = curve_bundle.get("best_test_sr")
                    result["test_paf"] = curve_bundle.get("best_test_paf")
                    result["test_ei"] = curve_bundle.get("best_test_ei")
                    result["results"]["test_sr"] = curve_bundle.get("best_test_sr")
                    result["results"]["test_paf"] = curve_bundle.get("best_test_paf")
                    result["results"]["test_ei"] = curve_bundle.get("best_test_ei")
                else:
                    result["test_mineral_detection"] = evaluator.evaluate_probability_map(
                        probability_map,
                        self.test_minerals_df,
                    )
                result["mineral_eval_available"] = bool(result["test_mineral_detection"])
            except Exception as exc:
                self.log_message.emit(f"矿点检测评估已跳过: {exc}")

        result["val_mineral_detection_rate"] = mean_val_mineral_detection_rate
        result["test_mineral_detection_rate"] = get_detection_rate(result["test_mineral_detection"])
        result["val_recall"] = float(result["results"].get("val_recall", result["val_mineral_detection_rate"] or 0.0) or 0.0)
        result["test_recall"] = float(result["results"].get("test_recall", result["test_mineral_detection_rate"] or 0.0) or 0.0)
        scoring_detection = float(result["val_mineral_detection_rate"] or result["val_recall"] or 0.0)
        if result.get("best_test_ei") is not None:
            result["composite_formula"] = "independent_test_ei"
            result["composite_score"] = float(result.get("best_test_ei") or 0.0)
        else:
            result["composite_formula"] = resolve_composite_formula(
                scoring_detection,
                result["results"].get("val_paf"),
                None,
            )
            result["composite_score"] = compute_composite_score(
                scoring_detection,
                result["results"].get("val_paf"),
                None,
            )
        result["metric_protocol"] = METRIC_PROTOCOL
        result["threshold_strategy"] = THRESHOLD_STRATEGY
        result["paf_scope"] = PAF_SCOPE
        result["threshold_rule"] = THRESHOLD_RULE
        result["primary_score"] = primary_test_ei_score(result)
        result["selection_strategy"] = "test_ei"
        result["search_boundary_hits"] = []
        result["improvement_advice"] = build_improvement_advice(result)
        return result

    def _select_wrapper_views(self, wrapper, train_loader, val_loader, test_loader, *, config=None):
        config = dict(config or {})
        return prepare_model_views(
            wrapper,
            train_loader,
            val_loader,
            test_loader,
            supervised_train_ratio=1.0,
            augmentation_enabled=bool(config.get("augmentation_enabled", False)),
            augmentation_noise_std=float(config.get("augmentation_noise_std", 0.01) or 0.01),
        )

    def _extract_labels(self, data):
        if data is None:
            return np.array([], dtype=np.int64)
        if isinstance(data, tuple):
            return (np.asarray(data[1]).reshape(-1) > 0).astype(np.int64)

        labels = []
        for _, batch_labels in data:
            batch = np.asarray(batch_labels.detach().cpu().numpy()).reshape(-1) > 0
            labels.append(batch.astype(np.int64))
        if not labels:
            return np.array([], dtype=np.int64)
        return np.concatenate(labels, axis=0)

    def _evaluate_on_test(self, model, test_loader, device):
        if isinstance(model, BaseModelWrapper):
            labels = self._extract_labels(test_loader)
            if len(labels) == 0:
                return {
                    "test_acc": 0.0,
                    "test_loss": 0.0,
                    "test_precision": 0.0,
                    "test_recall": 0.0,
                    "test_f1": 0.0,
                }

            labels = (np.asarray(labels).reshape(-1) > 0).astype(np.int64)
            predictions = (np.asarray(model.predict(test_loader)).reshape(-1) > 0).astype(np.int64)
            if len(predictions) != len(labels):
                raise ValueError("Prediction and label counts do not match.")

            precision, recall, f1, _ = precision_recall_fscore_support(
                labels,
                predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
            return {
                "test_acc": float(np.mean(predictions == labels)),
                "test_loss": 0.0,
                "test_precision": float(precision),
                "test_recall": float(recall),
                "test_f1": float(f1),
                "test_sr": float(compute_sr_paf_ei(labels, predictions).get("sr", 0.0)),
                "test_paf": float(compute_sr_paf_ei(labels, predictions).get("paf", 0.0)),
                "test_ei": float(compute_sr_paf_ei(labels, predictions).get("ei", 0.0)),
            }

        model.eval()
        correct = 0
        total = 0
        test_loss_sum = 0.0
        all_predictions = []
        all_labels = []
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = (labels.to(device) > 0).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss_sum += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        if total == 0 or len(test_loader) == 0:
            return {
                "test_acc": 0.0,
                "test_loss": 0.0,
                "test_precision": 0.0,
                "test_recall": 0.0,
                "test_f1": 0.0,
            }

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="binary",
            pos_label=1,
            zero_division=0,
        )

        return {
            "test_acc": correct / total,
            "test_loss": test_loss_sum / len(test_loader),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "test_sr": float(compute_sr_paf_ei(all_labels, all_predictions).get("sr", 0.0)),
            "test_paf": float(compute_sr_paf_ei(all_labels, all_predictions).get("paf", 0.0)),
            "test_ei": float(compute_sr_paf_ei(all_labels, all_predictions).get("ei", 0.0)),
        }

    def save_results(self, output_path):
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(self.results), handle, indent=2, ensure_ascii=False)
        self.log_message.emit(f"结果已保存到: {output_path}")
