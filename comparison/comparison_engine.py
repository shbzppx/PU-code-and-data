"""Core execution engine for model comparison experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime

import torch
from PyQt5.QtCore import QObject, pyqtSignal
from sklearn.metrics import precision_recall_fscore_support

from .analysis_utils import (
    build_improvement_advice,
    compute_composite_score,
    compute_enrichment_index,
    compute_prediction_area_ratio,
    compute_validation_score,
    get_detection_rate,
    to_jsonable,
)
from .mineral_evaluator import MineralEvaluator


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
        self.val_minerals_df = None
        self.test_minerals_df = None

    def add_experiment(self, model_type, model_class, config, experiment_name=None):
        if experiment_name is None:
            experiment_name = f"{model_type}_{len(self.experiments)}"

        experiment = {
            "id": len(self.experiments),
            "name": experiment_name,
            "model_type": model_type,
            "model_class": model_class,
            "config": config.copy(),
            "status": "pending",
        }
        self.experiments.append(experiment)
        self.log_message.emit(f"添加实验: {experiment_name}")

    def clear_experiments(self):
        self.experiments = []
        self.results = []

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
        dataset_summary=None,
        h5_path=None,
        val_minerals_df=None,
        test_minerals_df=None,
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
        self.val_minerals_df = val_minerals_df
        self.test_minerals_df = test_minerals_df

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
            self.experiment_completed.emit(self.results)
            self.log_message.emit(f"\n全部实验结束，成功 {len(self.results)} 个。")

    def _run_single_experiment(self, experiment, train_loader, val_loader, test_loader, experiment_id):
        from ..cnn.trainer import Trainer

        config = experiment["config"]
        model = experiment["model_class"](
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

        result = {
            "experiment_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{experiment_id}",
            "model_type": experiment["model_type"],
            "config": config,
            "dataset_summary": dict(self.current_dataset_summary or {}),
            "results": {
                "train_acc": max(train_acc_history) if train_acc_history else 0.0,
                "training_time_seconds": training_time,
            },
            "test_metrics_available": False,
            "mineral_eval_available": False,
            "val_mineral_detection": None,
            "test_mineral_detection": None,
            "val_prediction_area_ratio": None,
            "test_prediction_area_ratio": None,
        }

        if val_loader is not None:
            val_metrics = self._evaluate_loader(model, val_loader, device, prefix="val")
            result["results"].update(val_metrics)

        if test_loader is not None:
            test_metrics = self._evaluate_loader(model, test_loader, device, prefix="test")
            result["results"].update(test_metrics)
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
                    prediction_batch_size=config.get("prediction_batch_size", 128),
                )
                probability_map = evaluator.generate_probability_map_for_torch_model(
                    model,
                    device,
                    config,
                )
                result["val_prediction_area_ratio"] = compute_prediction_area_ratio(
                    probability_map,
                    threshold=evaluator.detection_threshold,
                )
                result["test_prediction_area_ratio"] = result["val_prediction_area_ratio"]
                result["val_mineral_detection"] = evaluator.evaluate_probability_map(
                    probability_map,
                    self.val_minerals_df,
                )
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
        result["val_ei"] = compute_enrichment_index(
            result["val_mineral_detection_rate"],
            result.get("val_prediction_area_ratio"),
        )
        result["test_ei"] = compute_enrichment_index(
            result["test_mineral_detection_rate"],
            result.get("test_prediction_area_ratio"),
        )
        result["validation_score"] = compute_validation_score(
            result["results"].get("val_recall"),
            result.get("val_prediction_area_ratio"),
        )
        result["composite_score"] = compute_composite_score(
            result["results"].get("val_acc", 0.0),
            result["val_mineral_detection_rate"],
            result["results"].get("val_f1"),
        )
        result["search_boundary_hits"] = []
        result["improvement_advice"] = build_improvement_advice(result)

        self.current_trainer = None
        return result

    def _evaluate_loader(self, model, loader, device, prefix="val"):
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        all_predictions = []
        all_labels = []
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss_sum += loss.item()

                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        metric_prefix = "test" if prefix == "test" else "val"
        if total == 0 or len(loader) == 0:
            return {
                f"{metric_prefix}_acc": 0.0,
                f"{metric_prefix}_loss": 0.0,
                f"{metric_prefix}_precision": 0.0,
                f"{metric_prefix}_recall": 0.0,
                f"{metric_prefix}_f1": 0.0,
            }

        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels,
                all_predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        except ValueError:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )

        return {
            f"{metric_prefix}_acc": correct / total,
            f"{metric_prefix}_loss": loss_sum / len(loader),
            f"{metric_prefix}_precision": float(precision),
            f"{metric_prefix}_recall": float(recall),
            f"{metric_prefix}_f1": float(f1),
        }

    def _evaluate_on_test(self, model, test_loader, device):
        return self._evaluate_loader(model, test_loader, device, prefix="test")

    def save_results(self, output_path):
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(self.results), handle, indent=2, ensure_ascii=False)
        self.log_message.emit(f"结果已保存到: {output_path}")
