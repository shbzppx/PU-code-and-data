"""AutoML engine for classic ML and deep learning comparison."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.metrics import f1_score

from .analysis_utils import (
    DETECTION_THRESHOLD,
    build_improvement_advice,
    compute_composite_score,
    detect_search_boundary_hits,
    get_detection_rate,
)
from .mineral_evaluator import MineralEvaluator
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
        candidates = {float(value) for value in np.linspace(0.10, 0.90, 17)}
        candidates.add(float(DETECTION_THRESHOLD))
        return sorted(candidates)

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

    def register_model(self, name, wrapper):
        self.models[name] = wrapper
        self.log_message.emit(f"已注册模型: {name}")

    def clear_models(self):
        self.models.clear()
        self.results.clear()

    def run_automl(self, dataset_bundle, n_trials=50):
        self.is_running = True
        self.results = []

        evaluator = None
        if dataset_bundle.h5_path and (
            (dataset_bundle.val_minerals_df is not None and len(dataset_bundle.val_minerals_df) > 0)
            or (dataset_bundle.test_minerals_df is not None and len(dataset_bundle.test_minerals_df) > 0)
        ):
            evaluator = MineralEvaluator(
                dataset_bundle.h5_path,
                buffer_radius=dataset_bundle.dataset_summary.get("buffer_radius", 500),
                prediction_patch_size=dataset_bundle.dataset_summary.get("patch_size"),
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
                score_fn=lambda active_wrapper, params, history: self._score_trial(
                    active_wrapper,
                    val_data,
                    dataset_bundle.val_minerals_df,
                    evaluator,
                ),
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

                final_history = wrapper.train(train_data, val_data, **best_params)
                val_metrics = self._evaluate_classification(wrapper, val_data)
                test_metrics = self._evaluate_classification(wrapper, test_data, prefix="test")
                training_time = time.time() - start_time
                train_accuracy = float(final_history.get("train_acc", 0.0))

                val_detection = None
                test_detection = None
                best_detection_threshold = None
                threshold_search = []
                if evaluator is not None:
                    try:
                        probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
                        best_detection_threshold, val_detection, threshold_search = self._select_best_threshold(
                            evaluator,
                            probability_map,
                            dataset_bundle.val_minerals_df,
                        )
                        if dataset_bundle.test_minerals_df is not None and len(dataset_bundle.test_minerals_df) > 0:
                            test_detection = evaluator.evaluate_probability_map(
                                probability_map,
                                dataset_bundle.test_minerals_df,
                                threshold=best_detection_threshold,
                            )
                    except Exception as exc:
                        self.log_message.emit(f"{model_name} 的矿点评估已跳过: {exc}")

                val_detection_rate = get_detection_rate(val_detection)
                test_detection_rate = get_detection_rate(test_detection)
                composite_score = compute_composite_score(
                    val_metrics.get("val_accuracy", 0.0),
                    val_detection_rate,
                    val_metrics.get("val_f1", 0.0),
                )

                result = {
                    "model_name": model_name,
                    "model_type": model_name,
                    "best_params": best_params,
                    "best_score": float(best_score),
                    "val_accuracy": val_metrics.get("val_accuracy", 0.0),
                    "val_f1": val_metrics.get("val_f1", 0.0),
                    "test_accuracy": test_metrics.get("test_accuracy"),
                    "test_f1": test_metrics.get("test_f1"),
                    "training_time": training_time,
                    "optimization_history": optimizer.get_optimization_history(),
                    "dataset_summary": dict(dataset_bundle.dataset_summary or {}),
                    "val_mineral_detection": val_detection,
                    "test_mineral_detection": test_detection,
                    "val_mineral_detection_rate": val_detection_rate,
                    "test_mineral_detection_rate": test_detection_rate,
                    "best_detection_threshold": best_detection_threshold,
                    "threshold_search": threshold_search,
                    "composite_score": composite_score,
                    # Keep the final trained wrapper in memory so export can reuse the exact evaluated model.
                    "_trained_wrapper": wrapper,
                    "results": {
                        "train_acc": train_accuracy,
                        "val_acc": val_metrics.get("val_accuracy", 0.0),
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
        self.results.sort(key=lambda item: item.get("composite_score", 0.0), reverse=True)
        self.all_completed.emit(self.results)
        return self.results

    def _score_trial(self, wrapper, val_data, val_minerals_df, evaluator):
        val_metrics = self._evaluate_classification(wrapper, val_data)
        val_detection = None
        best_threshold = None
        if evaluator is not None and val_minerals_df is not None and len(val_minerals_df) > 0:
            probability_map = evaluator.generate_probability_map_for_wrapper(wrapper)
            best_threshold, val_detection, _ = self._select_best_threshold(
                evaluator,
                probability_map,
                val_minerals_df,
            )

        val_detection_rate = get_detection_rate(val_detection)
        composite_score = compute_composite_score(
            val_metrics.get("val_accuracy", 0.0),
            val_detection_rate,
            val_metrics.get("val_f1", 0.0),
        )
        return {
            "score": composite_score,
            "metrics": {
                "val_accuracy": val_metrics.get("val_accuracy", 0.0),
                "val_f1": val_metrics.get("val_f1", 0.0),
                "val_mineral_detection_rate": val_detection_rate or 0.0,
                "best_detection_threshold": best_threshold,
                "composite_score": composite_score,
            },
        }

    def _select_split_views(self, wrapper, dataset_bundle):
        if wrapper.data_mode == "loader":
            return dataset_bundle.train_loader, dataset_bundle.val_loader, dataset_bundle.test_loader
        return dataset_bundle.train_data_array, dataset_bundle.val_data_array, dataset_bundle.test_data_array

    def _extract_labels(self, data):
        if data is None:
            return np.array([], dtype=np.int64)
        if isinstance(data, tuple):
            return np.asarray(data[1])

        labels = []
        for _, batch_labels in data:
            labels.append(batch_labels.cpu().numpy())
        if not labels:
            return np.array([], dtype=np.int64)
        return np.concatenate(labels, axis=0)

    def _evaluate_classification(self, wrapper, data, prefix="val"):
        if data is None:
            return {
                f"{prefix}_accuracy": None,
                f"{prefix}_f1": None,
            }

        labels = self._extract_labels(data)
        if len(labels) == 0:
            return {
                f"{prefix}_accuracy": None,
                f"{prefix}_f1": None,
            }

        predictions = wrapper.predict(data)
        accuracy = float(np.mean(predictions == labels))
        f1 = float(f1_score(labels, predictions, average="weighted", zero_division=0))
        return {
            f"{prefix}_accuracy": accuracy,
            f"{prefix}_f1": f1,
        }

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
                    "综合评分": f"{item.get('composite_score', 0.0):.4f}",
                    "验证准确率": f"{item.get('val_accuracy', 0.0):.4f}",
                    "验证F1": f"{item.get('val_f1', 0.0):.4f}",
                    "内部验证矿点检出率": "" if item.get("val_mineral_detection_rate") is None else f"{item['val_mineral_detection_rate']:.4f}",
                    "测试准确率": "" if item.get("test_accuracy") is None else f"{item['test_accuracy']:.4f}",
                    "测试F1": "" if item.get("test_f1") is None else f"{item['test_f1']:.4f}",
                    "测试矿点检出率": "" if item.get("test_mineral_detection_rate") is None else f"{item['test_mineral_detection_rate']:.4f}",
                    "最优Trial分数": f"{item.get('best_score', 0.0):.4f}",
                    "训练时间(秒)": f"{item.get('training_time', 0.0):.1f}",
                    "最优参数": str(item.get("best_params", {})),
                    "改进建议": "；".join(item.get("improvement_advice", [])),
                }
                for item in self.results
            ]
        )
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
