import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem, QLabel,
    QTextEdit, QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# ============================================================
# 0. 全局配置
# ============================================================

# 是否强行把 threshold=0 的曲线起点修正为 SR=1, PAF=1
FORCE_FULL_AREA_AT_ZERO = False


# ============================================================
# 1. 字段名：与你的 txt 表头对应
# ============================================================

COL_FOLD = "fold"
COL_METHOD = "probability_method"
COL_THR = "threshold"
COL_HIT = "命中矿点数"
COL_DEP = "验证矿点数"
COL_CELL = "高潜力单元数"
COL_TOTAL = "验证区域单元数"

REQUIRED_COLS = {
    COL_FOLD, COL_METHOD, COL_THR,
    COL_HIT, COL_DEP, COL_CELL, COL_TOTAL
}

# 也支持直接的 SR/PAF 列
REQUIRED_COLS_DIRECT = {
    COL_FOLD, COL_METHOD, COL_THR, "SR", "PAF"
}


# ============================================================
# 2. 数据读取与五折 pooled 汇总
# ============================================================

def read_validation_table(path: Path, model_name: str) -> pd.DataFrame:
    """
    读取单个模型的五折验证统计表。
    支持两种格式：
    1. 原始格式：需要提供 命中矿点数、验证矿点数、高潜力单元数、验证区域单元数 四列，从中计算 SR/PAF
    2. 直接格式：直接提供 SR、PAF 两列
    """
    last_error = None

    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(path, sep="\t", encoding=enc)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"无法读取文件：{path}\n最后一次错误：{last_error}")

    # 检查格式
    has_direct = REQUIRED_COLS_DIRECT.issubset(set(df.columns))
    has_raw = REQUIRED_COLS.issubset(set(df.columns))
    
    if not (has_direct or has_raw):
        missing_direct = REQUIRED_COLS_DIRECT - set(df.columns)
        missing_raw = REQUIRED_COLS - set(df.columns)
        raise ValueError(
            f"{path} 格式不匹配。\n"
            f"直接格式缺少：{missing_direct}\n"
            f"原始格式缺少：{missing_raw}"
        )

    df = df.copy()
    df["model"] = model_name

    # 数值转换
    df[COL_THR] = pd.to_numeric(df[COL_THR], errors="coerce")
    df["SR"] = pd.to_numeric(df["SR"], errors="coerce")
    df["PAF"] = pd.to_numeric(df["PAF"], errors="coerce")
    
    # 如果是原始格式，也要转换其他列
    if has_raw:
        for c in [COL_HIT, COL_DEP, COL_CELL, COL_TOTAL]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[COL_THR, "SR", "PAF"])
    return df


def build_pooled_pa_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    对五折结果进行 pooled 汇总。

    如果原始数据已有 SR/PAF 列，则直接平均；
    否则在每个 model / probability_method / threshold 下：
    1. 先汇总命中矿点数、验证矿点数、高潜力单元数、验证区域单元数；
    2. 再计算 SR 和 PAF。

    这样比直接平均 5 折 SR/PAF 更适合 P-A 曲线。
    """
    group_cols = ["model", COL_METHOD, COL_THR]
    
    # 检查是否已有原始计数列
    has_raw_cols = all(col in df.columns for col in [COL_HIT, COL_DEP, COL_CELL, COL_TOTAL])
    
    if has_raw_cols:
        # 原始格式：先汇总计数再计算 SR/PAF
        pooled = (
            df.groupby(group_cols, as_index=False)
              .agg({
                  COL_HIT: "sum",
                  COL_DEP: "sum",
                  COL_CELL: "sum",
                  COL_TOTAL: "sum",
              })
        )

        pooled["SR"] = pooled[COL_HIT] / pooled[COL_DEP]
        pooled["PAF"] = pooled[COL_CELL] / pooled[COL_TOTAL]
        pooled["EI"] = np.where(pooled["PAF"] > 0, pooled["SR"] / pooled["PAF"], np.nan)
    else:
        # 直接格式：平均 SR/PAF
        pooled = (
            df.groupby(group_cols, as_index=False)
              .agg({
                  "SR": "mean",
                  "PAF": "mean",
              })
        )
        pooled["EI"] = np.where(pooled["PAF"] > 0, pooled["SR"] / pooled["PAF"], np.nan)

    if FORCE_FULL_AREA_AT_ZERO:
        # 只用于希望强制复现文献 P-A 曲线从全区开始的情形。
        # 注意：这会覆盖 threshold 最小值处的 SR/PAF，不再完全等同于原始统计表。
        first_idx = pooled.groupby(["model", COL_METHOD])[COL_THR].idxmin()
        pooled.loc[first_idx, "SR"] = 1.0
        pooled.loc[first_idx, "PAF"] = 1.0

    return pooled.sort_values(["model", COL_METHOD, COL_THR]).reset_index(drop=True)


# ============================================================
# 3. P-A 交点计算
# ============================================================

def find_pa_intersection(curve: pd.DataFrame) -> dict:
    """
    计算文献式 P-A 曲线交点。

    文献图中：
    - 左轴：Prediction rate / known orebodies occupied rate，对应你的 SR；
    - 右轴：Area / study-area occupied percentage，对应你的 PAF；
    - 右轴是反向的，因此视觉交点满足：

        SR = 1 - PAF

      等价于：

        SR + PAF = 1

    返回：
    - threshold, SR, PAF：线性插值得到的 P-A 交点；
    - nearest_*：离交点最近的实际离散阈值。
    """
    g = curve.sort_values(COL_THR).reset_index(drop=True)

    x = g[COL_THR].to_numpy(dtype=float)
    sr = g["SR"].to_numpy(dtype=float)
    paf = g["PAF"].to_numpy(dtype=float)

    diff = sr + paf - 1.0

    # 最近的实际离散阈值
    nearest_i = int(np.nanargmin(np.abs(diff)))
    nearest = {
        "nearest_threshold": x[nearest_i],
        "nearest_SR": sr[nearest_i],
        "nearest_PAF": paf[nearest_i],
        "nearest_gap_SR_plus_PAF_minus_1": diff[nearest_i],
    }

    # 线性插值寻找 diff=0 的交点
    eps = 1e-12
    for i in range(len(diff) - 1):
        d0, d1 = diff[i], diff[i + 1]

        if np.isnan(d0) or np.isnan(d1):
            continue

        if abs(d0) < eps:
            return {
                "intersection_type": "exact",
                "threshold": x[i],
                "SR": sr[i],
                "PAF": paf[i],
                **nearest,
            }

        if d0 * d1 < 0:
            w = -d0 / (d1 - d0)

            return {
                "intersection_type": "linear_interpolation",
                "threshold": x[i] + w * (x[i + 1] - x[i]),
                "SR": sr[i] + w * (sr[i + 1] - sr[i]),
                "PAF": paf[i] + w * (paf[i + 1] - paf[i]),
                **nearest,
            }

    # 如果没有跨零，则退回到最近离散点
    return {
        "intersection_type": "nearest_only_no_crossing",
        "threshold": nearest["nearest_threshold"],
        "SR": nearest["nearest_SR"],
        "PAF": nearest["nearest_PAF"],
        **nearest,
    }


def summarize_intersections(pa: pd.DataFrame) -> pd.DataFrame:
    """
    汇总每个模型、每种 probability_method 的 P-A 交点。
    """
    rows = []

    for (model, method), g in pa.groupby(["model", COL_METHOD], sort=False):
        result = find_pa_intersection(g)
        result["model"] = model
        result[COL_METHOD] = method

        # 文献中交点的 Y 值越高，模型越好；
        # 在这里 Y 值就是交点处的 SR。
        result["PA_intersection_Y"] = result["SR"]

        rows.append(result)

    summary = pd.DataFrame(rows)

    cols = [
        "model", COL_METHOD, "intersection_type",
        "threshold", "SR", "PAF", "PA_intersection_Y",
        "nearest_threshold", "nearest_SR", "nearest_PAF",
        "nearest_gap_SR_plus_PAF_minus_1",
    ]

    return summary[cols].sort_values(
        [COL_METHOD, "PA_intersection_Y"],
        ascending=[True, False]
    )


# ============================================================
# 4. 绘图：复现文献 Fig. 10 风格
# ============================================================

def setup_chinese_font():
    """
    尽量让中文标题正常显示。
    如果本机没有中文字体，图中可能出现方框，但不影响计算。
    """
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
        "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_one_pa_panel(ax, curve: pd.DataFrame, inter: dict, title: str):
    """
    按文献 Fig. 10 的形式画单个 P-A 曲线：

    x 轴：predictive probability threshold
    左 y 轴：known deposits occupied rate / SR
    右 y 轴：area occupied rate / PAF，并反向显示
    """
    curve = curve.sort_values(COL_THR)

    x = curve[COL_THR]
    sr = curve["SR"]
    paf = curve["PAF"]

    # 左轴：SR / Prediction rate
    line_sr, = ax.plot(
        x, sr,
        color="crimson",
        linewidth=2.0,
        label="Prediction rate / SR"
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predictive probability threshold")
    ax.set_ylabel("Percentage of known deposits / SR")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # 右轴：PAF / Area，关键是反向
    ax_area = ax.twinx()

    line_area, = ax_area.plot(
        x, paf,
        color="forestgreen",
        linewidth=2.0,
        label="Area / PAF"
    )

    ax_area.set_ylabel("Percentage of prediction area / PAF")
    ax_area.set_ylim(1.02, -0.02)

    # 交点：在左轴上画 y=SR；
    # 因为右轴反向，PAF 会在同一视觉高度。
    ax.scatter(
        inter["threshold"],
        inter["SR"],
        s=45,
        color="royalblue",
        edgecolor="black",
        linewidth=0.5,
        zorder=5
    )

    label = (
        f"Intersection\n"
        f"thr={inter['threshold']:.3f}\n"
        f"SR={inter['SR']:.1%}\n"
        f"PAF={inter['PAF']:.1%}"
    )

    ax.annotate(
        label,
        xy=(inter["threshold"], inter["SR"]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", linewidth=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
    )

    ax.legend(
        handles=[line_area, line_sr],
        labels=["Area / PAF", "Prediction rate / SR"],
        loc="lower left",
        fontsize=8,
        frameon=True,
    )


def plot_pa_panels(pa: pd.DataFrame, summary: pd.DataFrame, out_png: Path):
    """
    绘制 P-A 曲线面板。
    按 method（行） × model（列）排列。
    """
    setup_chinese_font()

    methods = sorted(pa[COL_METHOD].drop_duplicates())
    models = sorted(pa["model"].drop_duplicates())

    nrows = len(methods)
    ncols = len(models)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.6 * ncols, 4.8 * nrows),
        squeeze=False
    )

    for r, method in enumerate(methods):
        for c, model in enumerate(models):
            ax = axes[r][c]

            curve = pa[
                (pa["model"] == model) &
                (pa[COL_METHOD] == method)
            ]

            if curve.empty:
                ax.axis("off")
                continue

            inter_row = summary[
                (summary["model"] == model) &
                (summary[COL_METHOD] == method)
            ].iloc[0].to_dict()

            title = f"{model} - {method}"
            plot_one_pa_panel(ax, curve, inter_row, title)

    fig.suptitle(
        "P-A curves based on five-fold pooled validation data",
        fontsize=14,
        y=1.01
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. PyQt5 GUI - 计算线程
# ============================================================

class ComputeThread(QThread):
    """后台计算线程，避免界面卡顿"""
    finished = pyqtSignal(str)  # 成功信号
    error = pyqtSignal(str)     # 错误信号
    
    def __init__(self, file_paths, out_dir):
        super().__init__()
        self.file_paths = file_paths
        self.out_dir = Path(out_dir)
    
    def run(self):
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取所有文件
            dfs = []
            for fpath in self.file_paths:
                model_name = Path(fpath).stem.upper()
                df = read_validation_table(Path(fpath), model_name)
                dfs.append(df)
            
            raw = pd.concat(dfs, ignore_index=True)
            
            # 五折 pooled 后的 P-A 曲线数据
            pa = build_pooled_pa_table(raw)
            
            # P-A 交点表
            summary = summarize_intersections(pa)
            
            # 保存数据
            pa.to_csv(
                self.out_dir / "PA_curve_pooled_data.csv",
                index=False,
                encoding="utf-8-sig"
            )
            
            summary.to_csv(
                self.out_dir / "PA_intersection_summary.csv",
                index=False,
                encoding="utf-8-sig"
            )
            
            # 绘图
            plot_pa_panels(
                pa,
                summary,
                self.out_dir / "PA_curve_literature_style.png"
            )
            
            # 输出结果
            show = summary.copy()
            for col in ["SR", "PAF", "PA_intersection_Y", "nearest_SR", "nearest_PAF"]:
                show[col] = (show[col] * 100).round(2)
            
            show["threshold"] = show["threshold"].round(4)
            show["nearest_threshold"] = show["nearest_threshold"].round(4)
            show["nearest_gap_SR_plus_PAF_minus_1"] = (
                show["nearest_gap_SR_plus_PAF_minus_1"].round(6)
            )
            
            result_text = "\n=== P-A 交点汇总 ===\n\n" + show.to_string(index=False)
            result_text += f"\n\n输出文件夹：{self.out_dir.resolve()}"
            result_text += "\n\n生成的文件："
            result_text += "\n  - PA_curve_literature_style.png（P-A 曲线图）"
            result_text += "\n  - PA_intersection_summary.csv（交点表）"
            result_text += "\n  - PA_curve_pooled_data.csv（五折汇总曲线数据）"
            
            self.finished.emit(result_text)
        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# 6. PyQt5 GUI - 主窗口
# ============================================================

class PAAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.selected_files = []
        self.work_dir = Path.cwd()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("P-A 曲线分析工具")
        self.setGeometry(100, 100, 900, 700)
        
        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout()
        
        # 文件夹选择区
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("工作目录："))
        self.folder_label = QLabel("未选择")
        folder_layout.addWidget(self.folder_label)
        folder_btn = QPushButton("选择文件夹")
        folder_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(folder_btn)
        layout.addLayout(folder_layout)
        
        # 文件列表区
        layout.addWidget(QLabel("文件夹中的 .txt 文件："))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(self.file_list.MultiSelection)
        layout.addWidget(self.file_list)
        
        # 按钮区
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("刷新文件列表")
        refresh_btn.clicked.connect(self.refresh_file_list)
        btn_layout.addWidget(refresh_btn)
        
        btn_layout.addStretch()
        
        compute_btn = QPushButton("开始计算")
        compute_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        compute_btn.clicked.connect(self.start_compute)
        btn_layout.addWidget(compute_btn)
        
        layout.addLayout(btn_layout)
        
        # 结果输出区
        layout.addWidget(QLabel("计算结果："))
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        font = QFont("Courier")
        font.setPointSize(9)
        self.result_text.setFont(font)
        layout.addWidget(self.result_text)
        
        main_widget.setLayout(layout)
    
    def select_folder(self):
        """选择工作文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择包含验证数据的文件夹",
            str(self.work_dir)
        )
        if folder:
            self.work_dir = Path(folder)
            self.folder_label.setText(folder)
            self.result_text.append(f"已选择文件夹：{folder}")
            self.refresh_file_list()
    
    def refresh_file_list(self):
        """刷新文件列表"""
        self.file_list.clear()
        self.selected_files.clear()
        
        if not self.work_dir.exists():
            self.result_text.setText("✗ 文件夹不存在！")
            return
        
        txt_files = sorted(self.work_dir.glob("*.txt"))
        
        if not txt_files:
            self.result_text.setText("✗ 文件夹中没有 .txt 文件！")
            return
        
        for txt_file in txt_files:
            item = QListWidgetItem(txt_file.name)
            self.file_list.addItem(item)
        
        self.result_text.setText(f"✓ 发现 {len(txt_files)} 个 .txt 文件，请选择至少 2 个文件进行计算。")
    
    def start_compute(self):
        """开始计算"""
        # 获取选中的文件
        selected_items = self.file_list.selectedItems()
        
        if len(selected_items) < 2:
            QMessageBox.warning(self, "警告", "请至少选择 2 个文件！")
            return
        
        self.selected_files = [
            str(self.work_dir / item.text())
            for item in selected_items
        ]
        
        # 验证文件
        for fpath in self.selected_files:
            if not Path(fpath).exists():
                QMessageBox.critical(self, "错误", f"文件不存在：{fpath}")
                return
        
        # 创建输出目录
        out_dir = self.work_dir / "pa_curve_output"
        
        # 启动后台计算
        self.result_text.setText("⏳ 正在计算，请稍候...")
        self.compute_thread = ComputeThread(self.selected_files, out_dir)
        self.compute_thread.finished.connect(self.on_compute_finished)
        self.compute_thread.error.connect(self.on_compute_error)
        self.compute_thread.start()
    
    def on_compute_finished(self, result_text):
        """计算完成"""
        self.result_text.setText("✓ 计算完成！\n" + result_text)
        QMessageBox.information(
            self,
            "成功",
            "P-A 曲线分析完成！\n\n输出文件已保存到：\n" + str(self.work_dir / "pa_curve_output")
        )
    
    def on_compute_error(self, error_text):
        """计算出错"""
        self.result_text.setText("✗ 计算出错！\n\n" + error_text)
        QMessageBox.critical(self, "错误", f"计算过程中出错：\n{error_text}")


# ============================================================
# 7. 程序入口
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PAAnalysisApp()
    window.show()
    sys.exit(app.exec_())