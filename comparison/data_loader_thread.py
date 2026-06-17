"""Background workers for model comparison data preparation and execution."""

from collections import Counter
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from .analysis_utils import to_jsonable
from .batch_report_generator import BatchReportGenerator


class DataLoaderThread(QThread):
    """Build comparison datasets in the background."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, gui, h5_path, train_minerals, val_minerals, test_minerals, h5_mode, build_config, no_ore_minerals=None):
        super().__init__()
        self.gui = gui
        self.h5_path = h5_path
        self.train_minerals = train_minerals
        self.val_minerals = val_minerals
        self.test_minerals = test_minerals
        self.h5_mode = h5_mode
        self.build_config = build_config
        self.no_ore_minerals = no_ore_minerals

    def run(self):
        try:
            bundle = self.gui._build_dataset_bundle(
                h5_path=self.h5_path,
                train_minerals=self.train_minerals,
                val_minerals=self.val_minerals,
                test_minerals=self.test_minerals,
                no_ore_minerals=self.no_ore_minerals,
                h5_mode=self.h5_mode,
                build_config=self.build_config,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(bundle)
        except Exception as exc:
            self.error.emit(str(exc))


class ComparisonRunThread(QThread):
    """Run the comparison engine without blocking the GUI."""

    error = pyqtSignal(str)

    def __init__(self, engine, dataset_bundle):
        super().__init__()
        self.engine = engine
        self.dataset_bundle = dataset_bundle

    def run(self):
        try:
            self.engine.run_experiments(
                train_loader=self.dataset_bundle.train_loader,
                val_loader=self.dataset_bundle.val_loader,
                test_loader=self.dataset_bundle.test_loader,
                dataset_summary=self.dataset_bundle.dataset_summary,
                h5_path=self.dataset_bundle.h5_path,
                val_minerals_df=self.dataset_bundle.val_minerals_df,
                test_minerals_df=self.dataset_bundle.test_minerals_df,
            )
        except Exception as exc:
            self.error.emit(str(exc))

    def stop(self):
        self.engine.stop()


class AutoMLRunThread(QThread):
    """Run AutoML experiments in the background."""

    model_started = pyqtSignal(str)
    model_progress = pyqtSignal(str, int, int)
    model_completed = pyqtSignal(str, dict)
    all_completed = pyqtSignal(list)
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, engine, dataset_bundle, n_trials):
        super().__init__()
        self.engine = engine
        self.dataset_bundle = dataset_bundle
        self.n_trials = n_trials

        self.engine.model_started.connect(self.model_started.emit)
        self.engine.model_progress.connect(self.model_progress.emit)
        self.engine.model_completed.connect(self.model_completed.emit)
        self.engine.all_completed.connect(self.all_completed.emit)
        if hasattr(self.engine, "log_message"):
            self.engine.log_message.connect(self.log_message.emit)
        if hasattr(self.engine, "error_occurred"):
            self.engine.error_occurred.connect(self.error.emit)

    def run(self):
        try:
            self.engine.run_automl(self.dataset_bundle, n_trials=self.n_trials)
        except Exception as exc:
            self.error.emit(str(exc))

    def stop(self):
        if hasattr(self.engine, "stop"):
            self.engine.stop()


class WorkflowRunThread(QThread):
    """Run the stage-wise workflow orchestrator without blocking the GUI."""

    stage_changed = pyqtSignal(str)
    scheme_progress = pyqtSignal(int, int, str)
    model_progress = pyqtSignal(str, str, int, int)
    log_message = pyqtSignal(str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        orchestrator,
        dataset_request,
        selected_models,
        candidate_config,
        runtime_mode,
        output_root,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.dataset_request = dataset_request
        self.selected_models = selected_models
        self.candidate_config = candidate_config
        self.runtime_mode = runtime_mode
        self.output_root = output_root

        self.orchestrator.stage_changed.connect(self.stage_changed.emit)
        self.orchestrator.scheme_progress.connect(self.scheme_progress.emit)
        self.orchestrator.model_progress.connect(self.model_progress.emit)
        self.orchestrator.log_message.connect(self.log_message.emit)
        self.orchestrator.workflow_completed.connect(self.completed.emit)
        self.orchestrator.error_occurred.connect(self.error.emit)

    def run(self):
        try:
            self.orchestrator.run_workflow(
                self.dataset_request,
                self.selected_models,
                self.candidate_config,
                runtime_mode=self.runtime_mode,
                output_root=self.output_root,
            )
        except Exception as exc:
            self.error.emit(str(exc))

    def stop(self):
        if hasattr(self.orchestrator, "stop"):
            self.orchestrator.stop()


class WorkflowBatchRunThread(QThread):
    """Run repeated split workflows sequentially in the background."""

    run_started = pyqtSignal(int, int, int)
    run_completed = pyqtSignal(int, dict)
    stage_changed = pyqtSignal(str)
    scheme_progress = pyqtSignal(int, int, str)
    model_progress = pyqtSignal(str, str, int, int)
    log_message = pyqtSignal(str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        orchestrator,
        run_requests,
        selected_models,
        candidate_config,
        runtime_mode,
        output_root,
        base_seed,
        test_split_ratio,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.run_requests = list(run_requests or [])
        self.selected_models = list(selected_models or [])
        self.candidate_config = dict(candidate_config or {})
        self.runtime_mode = runtime_mode
        self.output_root = output_root
        self.base_seed = int(base_seed)
        self.test_split_ratio = float(test_split_ratio)
        self._running = True

        self.orchestrator.stage_changed.connect(self.stage_changed.emit)
        self.orchestrator.scheme_progress.connect(self.scheme_progress.emit)
        self.orchestrator.model_progress.connect(self.model_progress.emit)
        self.orchestrator.log_message.connect(self.log_message.emit)

    def stop(self):
        self._running = False
        if hasattr(self.orchestrator, "stop"):
            self.orchestrator.stop()

    def _resolve_batch_root(self) -> Path:
        base_root = Path(self.output_root) if self.output_root else Path.cwd() / "outputs" / "model_comparison"
        batch_root = base_root / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        batch_root.mkdir(parents=True, exist_ok=True)
        return batch_root

    def _build_batch_summary(self, batch_root: Path, run_summaries):
        run_summaries = list(run_summaries or [])
        run_rows = []
        model_records = []
        best_run_index = None
        best_run_score = float("-inf")
        best_run = None
        best_result = None
        best_model_name = ""
        top1_counts = Counter()

        for index, summary in enumerate(run_summaries, start=1):
            best_item = dict(summary.get("best_result") or {})
            best_item_score = float(best_item.get("composite_score") or 0.0)
            best_item_name = str(best_item.get("model_name") or "")
            if best_item_name:
                top1_counts[best_item_name] += 1

            mineral_split_summary = dict(summary.get("mineral_split_summary") or {})
            train_block = dict(mineral_split_summary.get("train") or {})
            test_block = dict(mineral_split_summary.get("test") or {})
            no_ore_block = dict(mineral_split_summary.get("no_ore") or {})
            run_rows.append(
                {
                    "run_index": index,
                    "split_seed": int(summary.get("split_seed") or self.base_seed + index - 1),
                    "is_best_run": False,
                    "best_model_name": best_item_name,
                    "best_composite_score": round(best_item_score, 4),
                    "best_val_accuracy": round(float(best_item.get("val_accuracy") or 0.0), 4),
                    "best_test_mineral_detection_rate": round(float(best_item.get("test_mineral_detection_rate") or 0.0), 4),
                    "train_mineral_count": int(train_block.get("count", 0) or 0),
                    "test_mineral_count": int(test_block.get("count", 0) or 0),
                    "no_ore_point_count": int(no_ore_block.get("count", 0) or 0),
                    "output_dir": str(summary.get("output_dir", "")),
                    "output_name": Path(str(summary.get("output_dir", ""))).name,
                }
            )

            if best_item_score > best_run_score:
                best_run_score = best_item_score
                best_run_index = index
                best_run = dict(summary)
                best_result = best_item
                best_model_name = best_item_name

            for result in summary.get("final_results") or []:
                model_records.append(
                    {
                        "run_index": index,
                        "model_name": str(result.get("model_name") or ""),
                        "composite_score": float(result.get("composite_score") or 0.0),
                        "val_accuracy": float(result.get("val_accuracy") or 0.0),
                        "test_mineral_detection_rate": float(result.get("test_mineral_detection_rate") or 0.0),
                        "training_time": float(result.get("training_time") or 0.0),
                    }
                )

        if best_run_index is not None:
            run_rows[best_run_index - 1]["is_best_run"] = True

        model_rows = []
        if model_records:
            model_df = pd.DataFrame(model_records)
            for model_name, group in model_df.groupby("model_name", sort=False):
                composite_scores = group["composite_score"]
                val_scores = group["val_accuracy"]
                test_scores = group["test_mineral_detection_rate"]
                mean_score = float(composite_scores.mean())
                std_score = float(composite_scores.std(ddof=0)) if len(composite_scores) > 1 else 0.0
                model_rows.append(
                    {
                        "model_name": model_name,
                        "run_count": int(len(group)),
                        "top1_count": int(top1_counts.get(model_name, 0)),
                        "mean_composite_score": round(mean_score, 4),
                        "std_composite_score": round(std_score, 4),
                        "mean_val_accuracy": round(float(val_scores.mean()), 4),
                        "mean_test_mineral_detection_rate": round(float(test_scores.mean()), 4),
                        "stability_score": round(mean_score - std_score, 4),
                    }
                )

            model_rows.sort(
                key=lambda item: (
                    -(float(item.get("stability_score") or 0.0)),
                    -(float(item.get("mean_composite_score") or 0.0)),
                    -(int(item.get("top1_count") or 0)),
                    str(item.get("model_name") or ""),
                )
            )

        best_model = model_rows[0] if model_rows else {}
        if best_run is None:
            best_run = {}
        if best_result is None:
            best_result = {}
        first_run_summary = dict(run_summaries[0] or {}) if run_summaries else {}
        first_dataset_summary = dict(first_run_summary.get("dataset_summary") or {})

        summary = {
            "batch_mode": True,
            "split_mode": str(first_dataset_summary.get("split_mode") or "single"),
            "split_count": len(run_summaries),
            "base_seed": self.base_seed,
            "runtime_mode": self.runtime_mode,
            "test_split_ratio": self.test_split_ratio,
            "selected_models": self.selected_models,
            "candidate_config": self.candidate_config,
            "negative_sampling_mode": first_dataset_summary.get("negative_sampling_mode", ""),
            "negative_sampling_applied": bool(first_dataset_summary.get("negative_sampling_applied", False)),
            "negative_distance_multiplier": first_dataset_summary.get("negative_distance_multiplier", ""),
            "negative_distance_radius": first_dataset_summary.get("negative_distance_radius", ""),
            "no_ore_active": bool(first_dataset_summary.get("no_ore_active", False)),
            "no_ore_point_count": int(first_dataset_summary.get("no_ore_point_count", 0) or 0),
            "no_ore_sample_count": int(first_dataset_summary.get("no_ore_sample_count", 0) or 0),
            "no_ore_conflict_count": int(first_dataset_summary.get("no_ore_conflict_count", 0) or 0),
            "spatial_region_active": bool(first_dataset_summary.get("spatial_region_active", False)),
            "spatial_region_train_bounds": dict(first_dataset_summary.get("spatial_region_train_bounds") or {}),
            "spatial_region_test_bounds": dict(first_dataset_summary.get("spatial_region_test_bounds") or {}),
            "spatial_region_buffer_distance": float(first_dataset_summary.get("spatial_region_buffer_distance", 0.0) or 0.0),
            "spatial_region_train_sample_count": int(first_dataset_summary.get("spatial_region_train_sample_count", 0) or 0),
            "spatial_region_test_sample_count": int(first_dataset_summary.get("spatial_region_test_sample_count", 0) or 0),
            "spatial_region_gray_sample_count": int(first_dataset_summary.get("spatial_region_gray_sample_count", 0) or 0),
            "spatial_region_outside_sample_count": int(first_dataset_summary.get("spatial_region_outside_sample_count", 0) or 0),
            "spatial_region_overlap_sample_count": int(first_dataset_summary.get("spatial_region_overlap_sample_count", 0) or 0),
            "spatial_region_train_no_ore_sample_count": int(first_dataset_summary.get("spatial_region_train_no_ore_sample_count", 0) or 0),
            "spatial_region_test_no_ore_sample_count": int(first_dataset_summary.get("spatial_region_test_no_ore_sample_count", 0) or 0),
            "output_dir": str(batch_root),
            "best_run_index": best_run_index or 0,
            "best_run": best_run,
            "best_result": best_result,
            "best_model": best_model,
            "run_summaries": run_summaries,
            "run_rows": run_rows,
            "model_rows": model_rows,
            "workflow_config": dict(run_summaries[0].get("workflow_config") or {}) if run_summaries else {},
            "trial_scheme_summary": dict(run_summaries[0].get("trial_scheme_summary") or {}) if run_summaries else {},
            "mineral_split_summary": dict(best_run.get("mineral_split_summary") or {}),
            "batch_runs_path": str(batch_root / "batch_runs.csv"),
            "batch_models_path": str(batch_root / "batch_model_stability.csv"),
            "batch_summary_json_path": str(batch_root / "batch_summary.json"),
            "batch_summary_xlsx_path": str(batch_root / "batch_summary.xlsx"),
            "batch_report_path": str(batch_root / "batch_report.pdf"),
            "report_path": str(batch_root / "batch_report.pdf"),
            "leaderboard_path": str(batch_root / "batch_runs.csv"),
            "all_trials_path": str(batch_root / "batch_model_stability.csv"),
        }

        pd.DataFrame(run_rows).to_csv(summary["batch_runs_path"], index=False, encoding="utf-8-sig")
        pd.DataFrame(model_rows).to_csv(summary["batch_models_path"], index=False, encoding="utf-8-sig")
        Path(summary["batch_summary_json_path"]).write_text(
            json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        batch_report_generator = BatchReportGenerator(summary)
        try:
            batch_report_generator.generate_excel_report(summary["batch_summary_xlsx_path"])
        except Exception as exc:
            summary["batch_excel_error"] = str(exc)
            summary["batch_summary_xlsx_path"] = ""
            self.log_message.emit(f"Excel 汇总未生成: {exc}")
        batch_report_generator.generate_pdf_report(summary["batch_report_path"])
        Path(summary["batch_summary_json_path"]).write_text(
            json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary

    def run(self):
        try:
            batch_root = self._resolve_batch_root()
            run_summaries = []
            total_runs = len(self.run_requests)
            for index, request in enumerate(self.run_requests, start=1):
                if not self._running:
                    break
                split_seed = int(request.get("split_seed") or (self.base_seed + index - 1))
                self.run_started.emit(index, total_runs, split_seed)
                run_output_root = batch_root / f"run_{index:02d}_seed_{split_seed}"
                self.log_message.emit(f"开始第 {index}/{total_runs} 轮连续划分，seed={split_seed}")
                summary = self.orchestrator.run_workflow(
                    request,
                    self.selected_models,
                    self.candidate_config,
                    runtime_mode=self.runtime_mode,
                    output_root=str(run_output_root),
                    use_timestamp_subdir=False,
                )
                summary = dict(summary)
                summary["split_round"] = index
                summary["split_seed"] = split_seed
                summary["batch_output_dir"] = str(batch_root)
                summary["batch_run_output_dir"] = str(run_output_root)
                run_summaries.append(summary)
                self.run_completed.emit(index, summary)

            batch_summary = self._build_batch_summary(batch_root, run_summaries)
            self.completed.emit(batch_summary)
        except Exception as exc:
            self.error.emit(str(exc))


class ManifestRebuildThread(QThread):
    """Rebuild prediction artifacts from a saved manifest in the background."""

    log_message = pyqtSignal(str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, artifact_manager, manifest_path, output_dir):
        super().__init__()
        self.artifact_manager = artifact_manager
        self.manifest_path = manifest_path
        self.output_dir = output_dir

    def run(self):
        try:
            self.log_message.emit(f"Loading manifest: {self.manifest_path}")
            result = self.artifact_manager.rebuild_from_manifest(self.manifest_path, self.output_dir)
            self.completed.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
