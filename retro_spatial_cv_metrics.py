from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


ROOT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = ROOT_DIR / "模型训练"
COMMON_DIR = ROOT_DIR / "common"
for path in (ROOT_DIR, TRAIN_DIR, COMMON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import load_dataset  # noqa: E402
from feature_channel_utils import parse_selected_channels  # noqa: E402
from spatial_cv_metrics import (  # noqa: E402
    build_spatial_cv_context,
    evaluate_spatial_cv_fold,
    summarize_spatial_cv_metrics,
)
from utils import instantiate_model  # noqa: E402


DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "模型训练"
    / "result"
    / "先验概率、窗口大小选择"
    / "样本0.2种子36-C"
    / "prior_0p1_window_7_sample_0p2_seed_36"
    / "pucnntransformer_20260427_182155"
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_normalization(path: Path) -> Dict[str, object]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _as_float(value, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _selected_channels(params: Dict[str, object]) -> Optional[List[int]]:
    value = params.get("selected_channels", "")
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return parse_selected_channels(str(value or ""))


def _rebuild_dataset(params: Dict[str, object]):
    return load_dataset(
        data_path=str(params.get("dataset") or ""),
        label_path=str(params.get("label_path") or ""),
        test_size=_as_float(params.get("test_size"), 0.2),
        sample_ratio=_as_float(params.get("sample_ratio"), 1.0),
        split_mode=str(params.get("split_mode") or "legacy"),
        patch_size=_as_int(params.get("patch_size"), _as_int(params.get("img_size"), 16)),
        patch_stride=_as_int(params.get("patch_stride"), _as_int(params.get("patch_size"), 16)),
        use_reflect_padding=bool(params.get("reflect_padding", True)),
        selected_channels=_selected_channels(params),
        buffer_radius=_as_float(params.get("buffer_radius"), 15.0),
        spatial_cluster_n_clusters=_as_int(params.get("spatial_cluster_n_clusters"), 10),
        spatial_cluster_train_ratio=_as_float(params.get("spatial_cluster_train_ratio"), 0.7),
        full_mineral_training=bool(params.get("full_mineral_training", False)),
        mineral_training_strategy=str(params.get("mineral_training_strategy") or "holdout"),
        spatial_cluster_cv_buffer_distance=_as_float(params.get("spatial_cv_buffer_distance"), 0.0),
        random_state=_as_int(params.get("spatial_random_state"), 42),
        spatial_cv_folds=_as_int(params.get("cv_folds"), 5),
        no_ore_path=params.get("no_ore_path") or None,
    )


def _load_fold_model(model_path: Path, model_name: str, prior: float, input_shape, device):
    payload = torch.load(str(model_path), map_location=device, weights_only=False)
    state_dict = payload
    payload_shape = None
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
        payload_shape = (
            int(payload.get("input_channels", input_shape[0])),
            int(payload.get("input_height", input_shape[1])),
            int(payload.get("input_width", input_shape[2])),
        )
    model = instantiate_model(model_name, prior, payload_shape or input_shape)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _predict_positive_scores(model, features: torch.Tensor, batch_size: int, device) -> np.ndarray:
    loader = DataLoader(TensorDataset(features, torch.zeros(len(features), dtype=torch.long)), batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            logits = model(data).view(-1)
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    if not scores:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(scores).astype(np.float64)


def _folds_from_normalization(
    normalization_params: Dict[str, object],
    area_normalization_params: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    folds = normalization_params.get("spatial_cv_folds") or []
    area_folds = []
    if isinstance(area_normalization_params, dict):
        area_folds = list(area_normalization_params.get("spatial_cv_folds") or [])
    normalized = []
    for index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            continue
        val_indices = np.asarray(fold.get("val_indices", []), dtype=np.int64)
        train_indices = np.asarray(fold.get("train_indices", []), dtype=np.int64)
        if len(val_indices) == 0:
            continue
        val_area_indices = fold.get("val_area_indices")
        if val_area_indices is None and index < len(area_folds) and isinstance(area_folds[index], dict):
            val_area_indices = area_folds[index].get("val_area_indices")
        normalized.append(
            {
                "fold": int(fold.get("fold", index)) + 1,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "val_area_indices": None if val_area_indices is None else np.asarray(val_area_indices, dtype=np.int64),
            }
        )
    return normalized


def _write_csv(path: Path, fold_metrics: Iterable[Dict[str, object]]) -> None:
    columns = [
        "fold",
        "threshold",
        "val_sr",
        "val_paf",
        "val_ei",
        "val_detected_count",
        "val_mineral_count",
        "high_potential_count",
        "val_area_count",
        "model_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in fold_metrics:
            writer.writerow({column: item.get(column, "") for column in columns})


def run(args) -> Dict[str, object]:
    model_dir = Path(args.model_dir).resolve()
    params_path = model_dir / "params.json"
    norm_path = model_dir / "normalization_params.pth"
    if not params_path.exists():
        raise FileNotFoundError(f"params.json not found: {params_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"normalization_params.pth not found: {norm_path}")

    params = _load_json(params_path)
    old_norm = _load_normalization(norm_path)
    old_folds = _folds_from_normalization(old_norm)
    if not old_folds:
        raise RuntimeError("No spatial_cv_folds were found in the old normalization_params.pth.")

    print(f"旧训练目录: {model_dir}")
    print(f"检测到空间CV折数: {len(old_folds)}")
    print("开始按旧参数重建训练样本空间上下文...")
    (x_train, y_train), _, rebuilt_prior, rebuilt_norm = _rebuild_dataset(params)
    folds = _folds_from_normalization(old_norm, rebuilt_norm)
    context = build_spatial_cv_context(rebuilt_norm)
    if not context:
        raise RuntimeError(
            "重建数据后仍缺少 train_positions/train_mineral_ids/train_mineral_positions，"
            "无法按面积口径补算 SR/PAF/EI。"
        )

    if len(context["train_positions"]) != len(x_train):
        raise RuntimeError(
            f"空间坐标数量与训练样本数量不一致: positions={len(context['train_positions'])}, x_train={len(x_train)}"
        )

    max_fold_index = max(int(np.max(fold["val_indices"])) for fold in folds if len(fold["val_indices"]))
    if max_fold_index >= len(x_train):
        raise RuntimeError(
            f"旧CV折索引超出重建训练集范围: max_val_index={max_fold_index}, train_size={len(x_train)}"
        )

    if args.dry_run:
        return {
            "model_dir": str(model_dir),
            "dry_run": True,
            "fold_count": len(folds),
            "train_size": int(len(x_train)),
            "train_position_count": int(len(context["train_positions"])),
            "train_mineral_count": int(len(context["train_mineral_positions"])),
            "rebuilt_prior": float(rebuilt_prior),
        }

    device = torch.device(args.device)
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float32)
    model_name = str(params.get("model") or "pucnntransformer")
    prior = _as_float(params.get("resolved_prior", params.get("manual_prior")), float(rebuilt_prior))
    input_shape = tuple(int(v) for v in x_train_tensor.shape[1:4])
    batch_size = max(1, int(args.batch_size))

    fold_metrics = []
    for fold in folds:
        fold_number = int(fold["fold"])
        model_path = model_dir / f"model_fold{fold_number}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Fold model not found: {model_path}")

        val_indices = np.asarray(fold["val_indices"], dtype=np.int64)
        area_indices = fold.get("val_area_indices")
        area_indices = None if area_indices is None else np.asarray(area_indices, dtype=np.int64)
        area_features = context.get("area_features")
        metric_features = x_train_tensor[val_indices]
        if area_indices is not None and area_features is not None:
            metric_features = torch.as_tensor(area_features[area_indices], dtype=torch.float32)
        print(f"计算第 {fold_number}/{len(folds)} 折: val_samples={len(val_indices)}")
        model = _load_fold_model(model_path, model_name, prior, input_shape, device)
        probabilities = _predict_positive_scores(model, metric_features, batch_size, device)
        metric = evaluate_spatial_cv_fold(
            probabilities,
            val_indices,
            context,
            area_indices=area_indices,
            threshold_step=float(args.threshold_step),
            fixed_threshold=args.fixed_threshold,
            distance_threshold=float(args.distance_threshold),
        )
        if metric is None:
            print(f"第 {fold_number} 折缺少有效矿点或面积，已跳过。")
            continue
        metric = dict(metric)
        metric["fold"] = fold_number
        metric["model_path"] = str(model_path)
        fold_metrics.append(metric)
        print(
            f"第 {fold_number} 折: SR={metric['val_sr']:.4f}, "
            f"PAF={metric['val_paf']:.4f}, EI={metric['val_ei']:.4f}, "
            f"threshold={metric['threshold']:.2f}"
        )

    if not fold_metrics:
        raise RuntimeError("No fold produced valid spatial SR/PAF/EI metrics.")

    summary = summarize_spatial_cv_metrics(fold_metrics)
    result = {
        "source_model_dir": str(model_dir),
        "metric_note": (
            "Retroactive spatial CV metrics computed from old fold models. "
            "SR uses validation-fold mineral hits, PAF uses high-potential area fraction "
            "inside the validation fold, and EI = SR / PAF."
        ),
        "model": model_name,
        "prior": float(prior),
        "fold_count": int(len(fold_metrics)),
        "threshold_step": float(args.threshold_step),
        "fixed_threshold": args.fixed_threshold,
        "distance_threshold": float(args.distance_threshold),
        "rebuilt_prior": float(rebuilt_prior),
        "train_size": int(len(x_train)),
        "train_positive_count": int(np.sum(np.asarray(y_train) > 0)),
        "fold_metrics": fold_metrics,
        **summary,
    }

    output_json = Path(args.output_json).resolve() if args.output_json else model_dir / "spatial_cv_retro_metrics.json"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else model_dir / "spatial_cv_retro_metrics.csv"
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=_json_default)
    _write_csv(output_csv, fold_metrics)

    print("\n补算完成")
    print(f"CV SR:  {result.get('cv_sr_mean_std')}")
    print(f"CV PAF: {result.get('cv_paf_mean_std')}")
    print(f"CV EI:  {result.get('cv_ei_mean_std')}")
    print(f"JSON: {output_json}")
    print(f"CSV:  {output_csv}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retroactively compute spatial CV SR/PAF/EI for an old training result directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Old training result directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for fold model inference")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Threshold scan step; ignored when fixed threshold is set")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Use one fixed confidence threshold instead of max-EI scan")
    parser.add_argument("--distance-threshold", type=float, default=4.0, help="Mineral hit distance threshold")
    parser.add_argument("--output-json", default="", help="Optional output JSON path")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Only rebuild/check metadata; do not load fold models")
    parser.add_argument("--gui", action="store_true", help="Open the PyQt5 parameter window")
    return parser


def launch_gui(default_args) -> None:
    try:
        from PyQt5.QtCore import QThread, pyqtSignal
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyQt5 is not available in the current Python environment.") from exc

    class SignalStream:
        def __init__(self, signal):
            self.signal = signal

        def write(self, text):
            if text:
                self.signal.emit(str(text))

        def flush(self):
            pass

    class MetricWorker(QThread):
        log = pyqtSignal(str)
        finished_ok = pyqtSignal(dict)
        failed = pyqtSignal(str)

        def __init__(self, run_args):
            super().__init__()
            self.run_args = run_args

        def run(self):
            stream = SignalStream(self.log)
            try:
                with redirect_stdout(stream), redirect_stderr(stream):
                    result = run(self.run_args)
                self.finished_ok.emit(result)
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))

    class RetroMetricWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.worker = None
            self.setWindowTitle("临时空间CV SR/PAF/EI 补算工具")
            self.resize(980, 720)

            root = QWidget(self)
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            location_group = QGroupBox("运算位置")
            location_layout = QGridLayout(location_group)
            self.model_dir_edit = QLineEdit(str(default_args.model_dir))
            self.output_json_edit = QLineEdit(str(default_args.output_json or ""))
            self.output_csv_edit = QLineEdit(str(default_args.output_csv or ""))
            model_btn = QPushButton("选择...")
            json_btn = QPushButton("选择...")
            csv_btn = QPushButton("选择...")
            model_btn.clicked.connect(self._choose_model_dir)
            json_btn.clicked.connect(self._choose_output_json)
            csv_btn.clicked.connect(self._choose_output_csv)
            location_layout.addWidget(QLabel("旧训练目录"), 0, 0)
            location_layout.addWidget(self.model_dir_edit, 0, 1)
            location_layout.addWidget(model_btn, 0, 2)
            location_layout.addWidget(QLabel("输出 JSON"), 1, 0)
            location_layout.addWidget(self.output_json_edit, 1, 1)
            location_layout.addWidget(json_btn, 1, 2)
            location_layout.addWidget(QLabel("输出 CSV"), 2, 0)
            location_layout.addWidget(self.output_csv_edit, 2, 1)
            location_layout.addWidget(csv_btn, 2, 2)
            layout.addWidget(location_group)

            params_group = QGroupBox("计算参数")
            params_layout = QFormLayout(params_group)
            self.device_combo = QComboBox()
            self.device_combo.addItem("cpu")
            if torch.cuda.is_available():
                self.device_combo.addItem("cuda")
            device_index = self.device_combo.findText(str(default_args.device))
            if device_index >= 0:
                self.device_combo.setCurrentIndex(device_index)

            self.batch_spin = QSpinBox()
            self.batch_spin.setRange(1, 100000)
            self.batch_spin.setValue(int(default_args.batch_size))

            self.threshold_step_spin = QDoubleSpinBox()
            self.threshold_step_spin.setDecimals(4)
            self.threshold_step_spin.setRange(0.0001, 1.0)
            self.threshold_step_spin.setSingleStep(0.01)
            self.threshold_step_spin.setValue(float(default_args.threshold_step))

            self.fixed_threshold_check = QCheckBox("使用固定阈值")
            self.fixed_threshold_spin = QDoubleSpinBox()
            self.fixed_threshold_spin.setDecimals(4)
            self.fixed_threshold_spin.setRange(0.0, 1.0)
            self.fixed_threshold_spin.setSingleStep(0.01)
            if default_args.fixed_threshold is not None:
                self.fixed_threshold_check.setChecked(True)
                self.fixed_threshold_spin.setValue(float(default_args.fixed_threshold))
            else:
                self.fixed_threshold_spin.setValue(0.5)
            self.fixed_threshold_spin.setEnabled(self.fixed_threshold_check.isChecked())
            self.fixed_threshold_check.toggled.connect(self.fixed_threshold_spin.setEnabled)
            fixed_row = QHBoxLayout()
            fixed_row.addWidget(self.fixed_threshold_check)
            fixed_row.addWidget(self.fixed_threshold_spin)
            fixed_row.addStretch(1)
            fixed_widget = QWidget()
            fixed_widget.setLayout(fixed_row)

            self.distance_spin = QDoubleSpinBox()
            self.distance_spin.setDecimals(4)
            self.distance_spin.setRange(0.0, 1000000.0)
            self.distance_spin.setSingleStep(1.0)
            self.distance_spin.setValue(float(default_args.distance_threshold))

            self.dry_run_check = QCheckBox("只检查数据，不加载模型")
            self.dry_run_check.setChecked(bool(default_args.dry_run))

            params_layout.addRow("设备", self.device_combo)
            params_layout.addRow("推理 Batch Size", self.batch_spin)
            params_layout.addRow("阈值扫描步长", self.threshold_step_spin)
            params_layout.addRow("固定阈值", fixed_widget)
            params_layout.addRow("矿点命中距离", self.distance_spin)
            params_layout.addRow("", self.dry_run_check)
            layout.addWidget(params_group)

            button_row = QHBoxLayout()
            self.run_btn = QPushButton("开始计算")
            self.clear_btn = QPushButton("清空日志")
            self.run_btn.clicked.connect(self._start)
            self.clear_btn.clicked.connect(self._clear_log)
            button_row.addWidget(self.run_btn)
            button_row.addWidget(self.clear_btn)
            button_row.addStretch(1)
            layout.addLayout(button_row)

            self.summary_label = QLabel("等待计算")
            self.summary_label.setWordWrap(True)
            layout.addWidget(self.summary_label)

            self.log_edit = QPlainTextEdit()
            self.log_edit.setReadOnly(True)
            layout.addWidget(self.log_edit, 1)

        def _choose_model_dir(self):
            path = QFileDialog.getExistingDirectory(self, "选择旧训练目录", self.model_dir_edit.text().strip())
            if path:
                self.model_dir_edit.setText(path)

        def _choose_output_json(self):
            path, _ = QFileDialog.getSaveFileName(self, "选择输出 JSON", self.output_json_edit.text().strip() or "spatial_cv_retro_metrics.json", "JSON (*.json)")
            if path:
                self.output_json_edit.setText(path)

        def _choose_output_csv(self):
            path, _ = QFileDialog.getSaveFileName(self, "选择输出 CSV", self.output_csv_edit.text().strip() or "spatial_cv_retro_metrics.csv", "CSV (*.csv)")
            if path:
                self.output_csv_edit.setText(path)

        def _args_from_ui(self):
            return argparse.Namespace(
                model_dir=self.model_dir_edit.text().strip(),
                device=self.device_combo.currentText(),
                batch_size=int(self.batch_spin.value()),
                threshold_step=float(self.threshold_step_spin.value()),
                fixed_threshold=float(self.fixed_threshold_spin.value()) if self.fixed_threshold_check.isChecked() else None,
                distance_threshold=float(self.distance_spin.value()),
                output_json=self.output_json_edit.text().strip(),
                output_csv=self.output_csv_edit.text().strip(),
                dry_run=self.dry_run_check.isChecked(),
                gui=True,
            )

        def _start(self):
            if self.worker is not None and self.worker.isRunning():
                QMessageBox.information(self, "正在运行", "当前计算尚未结束。")
                return
            model_dir = self.model_dir_edit.text().strip()
            if not model_dir:
                QMessageBox.warning(self, "缺少目录", "请选择旧训练结果目录。")
                return
            if not os.path.isdir(model_dir):
                QMessageBox.warning(self, "目录无效", "旧训练结果目录不存在。")
                return
            self.log_edit.clear()
            self.summary_label.setText("正在计算...")
            self.run_btn.setEnabled(False)
            self.worker = MetricWorker(self._args_from_ui())
            self.worker.log.connect(self._append_log)
            self.worker.finished_ok.connect(self._finished)
            self.worker.failed.connect(self._failed)
            self.worker.start()

        def _append_log(self, text):
            self.log_edit.moveCursor(self.log_edit.textCursor().End)
            self.log_edit.insertPlainText(text)
            self.log_edit.moveCursor(self.log_edit.textCursor().End)

        def _finished(self, result):
            self.run_btn.setEnabled(True)
            if result.get("dry_run"):
                self.summary_label.setText(
                    f"检查完成：fold={result.get('fold_count')}, "
                    f"train={result.get('train_size')}, minerals={result.get('train_mineral_count')}"
                )
            else:
                self.summary_label.setText(
                    f"完成：SR={result.get('cv_sr_mean_std')} | "
                    f"PAF={result.get('cv_paf_mean_std')} | "
                    f"EI={result.get('cv_ei_mean_std')}"
                )

        def _failed(self, message):
            self.run_btn.setEnabled(True)
            self.summary_label.setText("计算失败")
            QMessageBox.critical(self, "计算失败", message)

        def _clear_log(self):
            self.log_edit.clear()

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True
    window = RetroMetricWindow()
    window.show()
    if owns_app:
        app.exec_()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.gui or len(sys.argv) == 1:
        launch_gui(args)
        return
    result = run(args)
    if args.dry_run:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
