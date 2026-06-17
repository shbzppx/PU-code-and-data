"""Stage-wise AutoML orchestration for model comparison workflows."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from .artifact_manager import ArtifactManager
from .automl_engine import AutoMLEngine
from .dataset_builder import ComparisonDataBuilder
from .mineral_evaluator import MineralEvaluator
from .report_generator import ReportGenerator


@dataclass(frozen=True)
class DatasetScheme:
    patch_size: int
    patch_stride: int
    buffer_radius: float
    n_blocks: int
    negative_sampling_mode: str
    negative_distance_multiplier: float
    split_mode: str
    sampling_percentage: float
    balance_ratio: Optional[float]
    train_mineral_count: int
    val_mineral_count: int
    test_mineral_count: int
    cache_key: str


@dataclass(frozen=True)
class ExperimentCandidate:
    dataset_scheme: DatasetScheme
    model_name: str
    param_space: Dict[str, object]
    stage: str


class WorkflowOrchestrator(QObject):
    """Run the end-to-end model search, ranking, packaging, and rebuild flow."""

    stage_changed = pyqtSignal(str)
    scheme_progress = pyqtSignal(int, int, str)
    model_progress = pyqtSignal(str, str, int, int)
    log_message = pyqtSignal(str)
    workflow_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    STAGE1_MAX_SCHEMES = 16
    STAGE1_TRIALS = 8
    STAGE1_TOP_K = 3
    STAGE1_DEEP_EPOCHS = 12
    STAGE2_TRIALS = 30
    STAGE2_DEEP_EPOCHS = 50

    def __init__(
        self,
        data_builder: ComparisonDataBuilder,
        wrapper_factory: Callable[[str, Dict[str, int], str], object],
        *,
        max_cached_bundles: int = 4,
    ) -> None:
        super().__init__()
        self.data_builder = data_builder
        self.wrapper_factory = wrapper_factory
        self.max_cached_bundles = max_cached_bundles
        self._bundle_cache: "OrderedDict[str, object]" = OrderedDict()
        self._running = False
        self._current_engine: Optional[AutoMLEngine] = None

    def stop(self) -> None:
        self._running = False
        if self._current_engine is not None:
            self._current_engine.stop()

    def _result_dataset_params(self, scheme: DatasetScheme, bundle) -> Dict[str, object]:
        params = asdict(scheme)
        params.update(
            {
                "no_ore_active": bool(bundle.dataset_summary.get("no_ore_active", False)),
                "no_ore_point_count": int(bundle.dataset_summary.get("no_ore_point_count", 0) or 0),
                "no_ore_sample_count": int(bundle.dataset_summary.get("no_ore_sample_count", 0) or 0),
                "no_ore_conflict_count": int(bundle.dataset_summary.get("no_ore_conflict_count", 0) or 0),
                "spatial_region_active": bool(bundle.dataset_summary.get("spatial_region_active", False)),
                "spatial_region_train_bounds": dict(bundle.dataset_summary.get("spatial_region_train_bounds") or {}),
                "spatial_region_test_bounds": dict(bundle.dataset_summary.get("spatial_region_test_bounds") or {}),
                "spatial_region_buffer_distance": float(bundle.dataset_summary.get("spatial_region_buffer_distance", 0.0) or 0.0),
                "spatial_region_train_sample_count": int(bundle.dataset_summary.get("spatial_region_train_sample_count", 0) or 0),
                "spatial_region_test_sample_count": int(bundle.dataset_summary.get("spatial_region_test_sample_count", 0) or 0),
                "spatial_region_gray_sample_count": int(bundle.dataset_summary.get("spatial_region_gray_sample_count", 0) or 0),
                "spatial_region_outside_sample_count": int(bundle.dataset_summary.get("spatial_region_outside_sample_count", 0) or 0),
                "spatial_region_overlap_sample_count": int(bundle.dataset_summary.get("spatial_region_overlap_sample_count", 0) or 0),
                "spatial_region_train_no_ore_sample_count": int(bundle.dataset_summary.get("spatial_region_train_no_ore_sample_count", 0) or 0),
                "spatial_region_test_no_ore_sample_count": int(bundle.dataset_summary.get("spatial_region_test_no_ore_sample_count", 0) or 0),
            }
        )
        return params

    def _scheme_from_mapping(self, mapping: Dict[str, object]) -> DatasetScheme:
        allowed_keys = {field.name for field in fields(DatasetScheme)}
        filtered = {key: mapping[key] for key in allowed_keys if key in mapping}
        return DatasetScheme(**filtered)

    def run_workflow(
        self,
        dataset_request: Dict[str, object],
        selected_models: Sequence[str],
        candidate_config: Dict[str, Sequence[object]],
        *,
        runtime_mode: str = "practical",
        output_root: Optional[str] = None,
        use_timestamp_subdir: bool = True,
    ) -> Dict[str, object]:
        if not selected_models:
            raise ValueError("至少需要选择一个模型才能运行工作流。")

        self._running = True
        output_dir = self._resolve_output_dir(output_root, use_timestamp_subdir=use_timestamp_subdir)
        stage1_records: List[Dict[str, object]] = []
        stage2_results: List[Dict[str, object]] = []
        workflow_config = {
            "runtime_mode": runtime_mode,
            "stage1_trials": self.STAGE1_TRIALS,
            "stage1_epochs": self.STAGE1_DEEP_EPOCHS,
            "stage2_trials": self.STAGE2_TRIALS,
            "stage2_epochs": self.STAGE2_DEEP_EPOCHS,
        }

        try:
            mineral_split_summary = self._build_mineral_split_summary(
                dataset_request.get("train_minerals"),
                dataset_request.get("val_minerals"),
                dataset_request.get("test_minerals"),
                no_ore_minerals=dataset_request.get("no_ore_minerals"),
                split_mode=str(dataset_request.get("split_mode", "single")),
            )
            schemes = self._generate_schemes(dataset_request, candidate_config)
            if not schemes:
                raise ValueError("未生成有效的数据方案。")

            self.stage_changed.emit("stage1")
            self.log_message.emit("第1阶段（粗筛）: 扫描候选数据方案...")
            for scheme_index, scheme in enumerate(schemes, start=1):
                if not self._running:
                    break
                self.scheme_progress.emit(scheme_index, len(schemes), scheme.cache_key)
                bundle = self._get_bundle(dataset_request, scheme)
                results = self._run_automl_for_bundle(
                    bundle,
                    selected_models,
                    stage="stage1",
                    n_trials=self.STAGE1_TRIALS,
                )
                for item in results:
                    item["dataset_params"] = self._result_dataset_params(scheme, bundle)
                    item["dataset_meta"] = dict(bundle.dataset_meta or {})
                    item["dataset_summary"] = dict(bundle.dataset_summary or {})
                    item["h5_path"] = dataset_request["h5_path"]
                    item["stage"] = "stage1"
                    item["workflow_config"] = dict(workflow_config)
                    item["candidate"] = asdict(
                        ExperimentCandidate(
                            dataset_scheme=scheme,
                            model_name=str(item.get("model_name")),
                            param_space=dict(self._create_wrapper(str(item.get("model_name")), bundle.dataset_meta, "stage1").get_param_space()),
                            stage="stage1",
                        )
                    )

                scheme_score = float(np.mean([item.get("composite_score", 0.0) for item in results])) if results else 0.0
                stage1_records.append(
                    {
                        "scheme": asdict(scheme),
                        "scheme_score": scheme_score,
                        "results": results,
                    }
                )

            if not stage1_records:
                raise RuntimeError("第1阶段未产生任何有效候选方案。")

            top_stage1 = sorted(
                stage1_records,
                key=lambda item: item.get("scheme_score", 0.0),
                reverse=True,
            )[: self.STAGE1_TOP_K]

            self.stage_changed.emit("stage2")
            self.log_message.emit("第2阶段（精筛）: 对排名靠前的数据方案执行正式 AutoML...")
            for scheme_rank, stage1_item in enumerate(top_stage1, start=1):
                if not self._running:
                    break
                scheme = DatasetScheme(**stage1_item["scheme"])
                bundle = self._get_bundle(dataset_request, scheme)
                self.scheme_progress.emit(scheme_rank, len(top_stage1), scheme.cache_key)
                results = self._run_automl_for_bundle(
                    bundle,
                    selected_models,
                    stage="stage2",
                    n_trials=self.STAGE2_TRIALS,
                )
                for item in results:
                    item["dataset_params"] = self._result_dataset_params(scheme, bundle)
                    item["dataset_meta"] = dict(bundle.dataset_meta or {})
                    item["dataset_summary"] = dict(bundle.dataset_summary or {})
                    item["h5_path"] = dataset_request["h5_path"]
                    item["stage"] = "stage2"
                    item["workflow_config"] = dict(workflow_config)
                stage2_results.extend(results)

            if not stage2_results:
                raise RuntimeError("第2阶段未产生任何最终结果。")

            ranked_results = sorted(stage2_results, key=self._result_sort_key)
            packaged_results = self._package_top_results(
                ranked_results,
                dataset_request,
                output_dir,
                runtime_mode=runtime_mode,
            )
            trial_scheme_summary = self._build_trial_scheme_summary(stage1_records, top_stage1)
            for item in packaged_results:
                item["trial_scheme_summary"] = dict(trial_scheme_summary)
                item["mineral_split_summary"] = dict(mineral_split_summary)

            report_path = str(Path(output_dir) / "report.pdf")
            ReportGenerator(packaged_results).generate_pdf_report(report_path)
            export_stage_results = [
                {key: value for key, value in item.items() if key != "_trained_wrapper"}
                for item in (stage2_results + [record for group in stage1_records for record in group["results"]])
            ]
            export_paths = ArtifactManager(self.wrapper_factory).save_run_exports(
                output_dir,
                packaged_results,
                report_path=report_path,
                stage_results=export_stage_results,
            )

            summary = {
                "output_dir": output_dir,
                "stage1_schemes": [
                    {
                        "scheme": dict(record.get("scheme") or {}),
                        "scheme_score": float(record.get("scheme_score", 0.0) or 0.0),
                    }
                    for record in stage1_records
                ],
                "final_results": packaged_results,
                "best_result": packaged_results[0] if packaged_results else None,
                "best_manifest_path": packaged_results[0].get("rebuild_manifest_path", "") if packaged_results else "",
                "best_model_path": packaged_results[0].get("model_artifact_path", "") if packaged_results else "",
                "workflow_config": workflow_config,
                "trial_scheme_summary": trial_scheme_summary,
                "mineral_split_summary": mineral_split_summary,
            }
            summary.update(export_paths)
            self.workflow_completed.emit(summary)
            return summary
        finally:
            self._current_engine = None
            self._running = False
            self._release_bundle_cache()

    def _package_top_results(
        self,
        ranked_results: Sequence[Dict[str, object]],
        dataset_request: Dict[str, object],
        output_dir: str,
        *,
        runtime_mode: str,
    ) -> List[Dict[str, object]]:
        artifact_manager = ArtifactManager(self.wrapper_factory)
        packaged_results: List[Dict[str, object]] = []
        seen_models = set()
        best_results: List[Dict[str, object]] = []
        for result in ranked_results:
            model_name = str(result.get("model_name") or "")
            if model_name in seen_models:
                continue
            seen_models.add(model_name)
            best_results.append(result)

        # Preserve only one winner per model, then materialize those winners.
        for rank, result in enumerate(sorted(best_results, key=self._result_sort_key), start=1):
            if not self._running:
                break
            scheme = self._scheme_from_mapping(dict(result.get("dataset_params") or {}))
            bundle = self._get_bundle(dataset_request, scheme)
            wrapper = result.get("_trained_wrapper")
            if wrapper is None:
                raise RuntimeError(f"缺少用于导出的已训练模型实例: {result.get('model_name')}")
            self.log_message.emit(
                f"Packaging best result for model {result.get('model_name')} using evaluated checkpoint: {scheme.cache_key}"
            )
            evaluator = MineralEvaluator(
                bundle.h5_path,
                buffer_radius=float(bundle.dataset_summary.get("buffer_radius", 500)),
                prediction_patch_size=int(bundle.dataset_summary.get("patch_size", bundle.dataset_meta.get("image_size", 1))),
            )
            package_dir = Path(output_dir) / ("best_run" if rank == 1 else f"candidate_runs/rank_{rank:02d}_{self._safe_name(str(result.get('model_name')))}")
            package_data = artifact_manager.package_trained_result(
                result,
                wrapper,
                evaluator,
                output_dir=str(package_dir),
                val_minerals_df=bundle.val_minerals_df,
                test_minerals_df=bundle.test_minerals_df,
                runtime_mode=runtime_mode,
            )
            enriched = {key: value for key, value in result.items() if key != "_trained_wrapper"}
            enriched.update(package_data)
            packaged_results.append(enriched)

        return sorted(packaged_results, key=self._result_sort_key)

    def _build_trial_scheme_summary(
        self,
        stage1_records: Sequence[Dict[str, object]],
        top_stage1: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        """Build a compact overview of the dataset schemes explored by the workflow."""

        stage1_trials = []
        for index, record in enumerate(stage1_records, start=1):
            stage1_trials.append(
                {
                    "rank": index,
                    "scheme": dict(record.get("scheme") or {}),
                    "scheme_score": float(record.get("scheme_score", 0.0) or 0.0),
                }
            )

        return {
            "stage1_total": len(stage1_trials),
            "stage2_total": len(top_stage1),
            "stage1_trials": stage1_trials,
        }

    def _build_mineral_split_summary(
        self,
        train_minerals: Optional[pd.DataFrame],
        val_minerals: Optional[pd.DataFrame],
        test_minerals: Optional[pd.DataFrame] = None,
        no_ore_minerals: Optional[pd.DataFrame] = None,
        *,
        split_mode: str = "single",
        preview_limit: int = 6,
    ) -> Dict[str, object]:
        effective_split_mode = str(split_mode or "").strip().lower() or "single"
        if effective_split_mode in {"single", "spatial_region"}:
            preview_limit = None
        summary = {
            "split_mode": effective_split_mode,
            "train": self._build_mineral_preview(train_minerals, preview_limit=preview_limit),
            "val": self._build_mineral_preview(val_minerals, preview_limit=preview_limit),
            "test": self._build_mineral_preview(test_minerals, preview_limit=preview_limit),
        }
        if no_ore_minerals is not None and len(no_ore_minerals) > 0:
            summary["no_ore"] = self._build_mineral_preview(no_ore_minerals, preview_limit=preview_limit)
        return summary

    def _build_mineral_preview(
        self,
        minerals_df: Optional[pd.DataFrame],
        *,
        preview_limit: Optional[int] = 6,
    ) -> Dict[str, object]:
        if minerals_df is None or len(minerals_df) == 0:
            return {"count": 0, "items": [], "truncated": False}

        frame = minerals_df.reset_index(drop=True)
        if preview_limit is None:
            preview_frame = frame
        else:
            preview_frame = frame.head(int(preview_limit))
        items = [
            self._format_mineral_preview_row(index + 1, row)
            for index, row in preview_frame.iterrows()
        ]
        return {
            "count": int(len(frame)),
            "items": items,
            "truncated": False if preview_limit is None else int(len(frame)) > int(preview_limit),
        }

    def _format_mineral_preview_row(self, index: int, row: pd.Series) -> str:
        descriptor_columns = [
            "name",
            "矿点名称",
            "sample_name",
            "id",
            "编号",
            "label",
            "label_id",
            "class",
            "category",
            "type",
        ]

        descriptor_column = None
        descriptor = None
        for column in descriptor_columns:
            if column in row.index and self._is_useful_value(row[column]):
                descriptor_column = column
                descriptor = row[column]
                break

        parts = [f"{index}."]
        if descriptor_column is not None:
            parts.append(f"{descriptor_column}={self._format_mineral_value(descriptor)}")

        if "x" in row.index and "y" in row.index:
            parts.append(
                f"x={self._format_mineral_value(row['x'])}, y={self._format_mineral_value(row['y'])}"
            )
        else:
            coordinate_columns = [
                "coord_x",
                "coord_y",
                "point_x",
                "point_y",
                "east",
                "north",
                "easting",
                "northing",
            ]
            detected = []
            for column in coordinate_columns:
                if column in row.index and self._is_useful_value(row[column]):
                    detected.append(f"{column}={self._format_mineral_value(row[column])}")
            if detected:
                parts.extend(detected[:2])

        return " | ".join(parts)

    def _format_mineral_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if abs(numeric - round(numeric)) < 1e-9:
                return str(int(round(numeric)))
            return f"{numeric:.4g}"
        text = str(value).strip()
        return text or ""

    def _is_useful_value(self, value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, (float, np.floating)) and np.isnan(float(value)):
            return False
        return str(value).strip() != ""

    def _run_automl_for_bundle(
        self,
        bundle,
        selected_models: Sequence[str],
        *,
        stage: str,
        n_trials: int,
    ) -> List[Dict[str, object]]:
        engine = AutoMLEngine()
        self._current_engine = engine
        engine.log_message.connect(self.log_message.emit)
        engine.model_progress.connect(
            lambda model_name, current, total, stage_name=stage: self.model_progress.emit(stage_name, model_name, current, total)
        )
        for model_name in selected_models:
            engine.register_model(model_name, self._create_wrapper(str(model_name), bundle.dataset_meta, stage))
        results = engine.run_automl(bundle, n_trials=n_trials)
        return [dict(item) for item in results]

    def _create_wrapper(self, model_name: str, dataset_meta: Dict[str, int], stage: str):
        return self.wrapper_factory(
            model_name,
            dataset_meta,
            stage,
        )

    def _generate_schemes(
        self,
        dataset_request: Dict[str, object],
        candidate_config: Dict[str, Sequence[object]],
    ) -> List[DatasetScheme]:
        patch_sizes = self._normalize_int_candidates(candidate_config.get("patch_size_candidates") or [dataset_request["build_config"]["patch_size"]])
        patch_strides = self._normalize_int_candidates(candidate_config.get("patch_stride_candidates") or [dataset_request["build_config"]["patch_stride"]])
        buffer_radii = self._normalize_float_candidates(candidate_config.get("buffer_radius_candidates") or [dataset_request["build_config"]["buffer_radius"]])
        sampling_percentages = self._normalize_float_candidates(candidate_config.get("sampling_percentage_candidates") or [1.0])
        balance_ratio_candidates = candidate_config.get("balance_ratio_candidates") or [None]
        n_blocks = int(dataset_request["build_config"].get("n_blocks", 3))
        negative_sampling_mode = str(dataset_request["build_config"].get("negative_sampling_mode", "default") or "default").strip().lower() or "default"
        negative_distance_multiplier = float(dataset_request["build_config"].get("negative_distance_multiplier", 2.0) or 2.0)
        region_hash = self._hash_region_split(dataset_request["build_config"].get("spatial_region_split"))

        mineral_hash = self._hash_mineral_split(
            dataset_request["train_minerals"],
            dataset_request["val_minerals"],
            dataset_request["test_minerals"],
            dataset_request.get("no_ore_minerals"),
        )
        schemes: List[DatasetScheme] = []
        for patch_size, patch_stride, buffer_radius, sampling_percentage, balance_ratio in product(
            patch_sizes,
            patch_strides,
            buffer_radii,
            sampling_percentages,
            balance_ratio_candidates,
        ):
            balance_ratio_value = None if balance_ratio is None else float(balance_ratio)
            cache_key = (
                f"{Path(str(dataset_request['h5_path'])).stem}"
                f"|{dataset_request.get('split_mode', 'single')}"
                f"|ps={int(patch_size)}|st={int(patch_stride)}|br={float(buffer_radius):.3f}"
                f"|nmode={negative_sampling_mode}|ndm={negative_distance_multiplier:.3f}"
                f"|sp={float(sampling_percentage):.3f}|bal={balance_ratio_value if balance_ratio_value is not None else 'default'}"
                f"|nb={int(n_blocks)}|region={region_hash}|{mineral_hash}"
            )
            schemes.append(
                DatasetScheme(
                    patch_size=int(patch_size),
                    patch_stride=int(patch_stride),
                    buffer_radius=float(buffer_radius),
                    n_blocks=int(n_blocks),
                    negative_sampling_mode=negative_sampling_mode,
                    negative_distance_multiplier=float(negative_distance_multiplier),
                    split_mode=str(dataset_request.get("split_mode", "single")),
                    sampling_percentage=float(sampling_percentage),
                    balance_ratio=balance_ratio_value,
                    train_mineral_count=int(len(dataset_request["train_minerals"])),
                    val_mineral_count=int(len(dataset_request["val_minerals"])),
                    test_mineral_count=int(len(dataset_request["test_minerals"])),
                    cache_key=cache_key,
                )
            )

        if len(schemes) > self.STAGE1_MAX_SCHEMES:
            self.log_message.emit(
                f"已生成 {len(schemes)} 组数据方案，Stage 1 仅保留前 {self.STAGE1_MAX_SCHEMES} 组进行筛选。"
            )
            schemes = schemes[: self.STAGE1_MAX_SCHEMES]
        return schemes

    def _get_bundle(self, dataset_request: Dict[str, object], scheme: DatasetScheme):
        if scheme.cache_key in self._bundle_cache:
            bundle = self._bundle_cache.pop(scheme.cache_key)
            self._bundle_cache[scheme.cache_key] = bundle
            return bundle

        build_config = dict(dataset_request["build_config"])
        build_config.update(
            {
                "patch_size": int(scheme.patch_size),
                "patch_stride": int(scheme.patch_stride),
                "buffer_radius": float(scheme.buffer_radius),
                "n_blocks": int(scheme.n_blocks),
                "negative_sampling_mode": str(scheme.negative_sampling_mode),
                "negative_distance_multiplier": float(scheme.negative_distance_multiplier),
                "sampling_percentage": float(scheme.sampling_percentage),
            }
        )
        if scheme.balance_ratio is not None:
            build_config["balance_ratio"] = float(scheme.balance_ratio)
        bundle = self.data_builder.build_bundle(
            h5_path=dataset_request["h5_path"],
            train_minerals=dataset_request["train_minerals"],
            val_minerals=dataset_request["val_minerals"],
            test_minerals=dataset_request["test_minerals"],
            no_ore_minerals=dataset_request.get("no_ore_minerals"),
            h5_mode=dataset_request["h5_mode"],
            build_config=build_config,
            progress_callback=self.log_message.emit,
        )
        bundle.dataset_summary["cache_key"] = scheme.cache_key
        self._bundle_cache[scheme.cache_key] = bundle
        while len(self._bundle_cache) > self.max_cached_bundles:
            old_key, old_bundle = self._bundle_cache.popitem(last=False)
            self.log_message.emit(f"释放缓存的数据方案: {old_key}")
            self.data_builder.release_bundle(old_bundle)
        return bundle

    def _release_bundle_cache(self) -> None:
        while self._bundle_cache:
            _, bundle = self._bundle_cache.popitem(last=False)
            self.data_builder.release_bundle(bundle)

    def _normalize_int_candidates(self, values: Iterable[object]) -> List[int]:
        unique_values = sorted({int(float(value)) for value in values})
        if not unique_values:
            raise ValueError("Candidate list cannot be empty.")
        return unique_values

    def _normalize_float_candidates(self, values: Iterable[object]) -> List[float]:
        unique_values = sorted({float(value) for value in values})
        if not unique_values:
            raise ValueError("Candidate list cannot be empty.")
        return unique_values

    def _hash_mineral_split(self, *mineral_dfs) -> str:
        hashed_parts: List[str] = []
        for mineral_df in mineral_dfs:
            if mineral_df is None or len(mineral_df) == 0:
                hashed_parts.append("empty")
                continue
            frame = mineral_df.copy().sort_values(by=["x", "y"]).reset_index(drop=True)
            hash_value = pd.util.hash_pandas_object(frame, index=True).sum()
            hashed_parts.append(f"{int(hash_value):x}")
        return "_".join(hashed_parts)

    def _hash_region_split(self, region_config: Optional[Dict[str, object]]) -> str:
        if not region_config:
            return "none"
        train_region = dict(region_config.get("train_region") or {})
        test_region = dict(region_config.get("test_region") or {})
        buffer_distance = float(region_config.get("buffer_distance", 0.0) or 0.0)
        parts = [
            train_region.get("xmin", ""),
            train_region.get("xmax", ""),
            train_region.get("ymin", ""),
            train_region.get("ymax", ""),
            test_region.get("xmin", ""),
            test_region.get("xmax", ""),
            test_region.get("ymin", ""),
            test_region.get("ymax", ""),
            buffer_distance,
        ]
        return "_".join(f"{float(value):.6f}" if value != "" else "-" for value in parts)

    def _result_sort_key(self, result: Dict[str, object]):
        return (
            -(float(result.get("composite_score") or 0.0)),
            -(float(result.get("val_f1") or 0.0)),
            -(float(result.get("test_mineral_detection_rate") or 0.0)),
            float(result.get("training_time") or 0.0),
            str(result.get("model_name") or ""),
        )

    def _resolve_output_dir(self, output_root: Optional[str], *, use_timestamp_subdir: bool = True) -> str:
        if output_root:
            root = Path(output_root)
        else:
            root = Path.cwd() / "outputs" / "model_comparison"
        target = root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f") if use_timestamp_subdir else root
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    def _safe_name(self, value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value).strip("_") or "model"
