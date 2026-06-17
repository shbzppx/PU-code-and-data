import os
import queue
import threading
import time
import sys

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QAbstractItemView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QPlainTextEdit,
)

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

try:
    from feature.grid_to_h5 import combine_grid_files_to_h5
except ImportError:
    from grid_to_h5 import combine_grid_files_to_h5


PAD_MODE_OPTIONS = [
    ("边缘填充", "edge"),
    ("常量填充", "constant"),
    ("镜像填充", "reflect"),
    ("对称填充", "symmetric"),
    ("环绕填充", "wrap"),
]
PAD_LABEL_TO_VALUE = {label: value for label, value in PAD_MODE_OPTIONS}


def _read_tabular_input_file(input_file):
    """自动识别并读取 CSV / Excel / DAT / TXT 表格文件。"""

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件: {input_file}")

    ext = os.path.splitext(input_file)[1].lower()
    if ext in {".csv"}:
        return pd.read_csv(input_file, encoding="utf-8-sig")
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(input_file)
    if ext in {".dat", ".txt"}:
        try:
            return pd.read_csv(input_file, sep=r"\s+|,|;|\t", engine="python")
        except Exception:
            return pd.read_csv(input_file, sep=r"\s+", engine="python")

    try:
        return pd.read_csv(input_file, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_excel(input_file)
        except Exception:
            return pd.read_csv(input_file, sep=r"\s+|,|;|\t", engine="python")


def _detect_column_by_options(df, preferred_name, aliases, fallback_index=None):
    candidates = [preferred_name] + list(aliases)
    for candidate in candidates:
        for col in df.columns:
            if str(col).strip() == str(candidate).strip():
                return col
    if fallback_index is not None and len(df.columns) > fallback_index:
        return df.columns[fallback_index]
    raise ValueError(f"无法识别字段: {preferred_name}")


def _normalize_spatial_input(df, band_col, x_col, y_col, use_header=True):
    """将输入表标准化为 band/x/y 三列。"""

    if use_header:
        x_real = _detect_column_by_options(df, x_col, ["X", "x", "lon", "longitude", "经度"], 1)
        y_real = _detect_column_by_options(df, y_col, ["Y", "y", "lat", "latitude", "纬度"], 2)

        if band_col:
            band_real = _detect_column_by_options(df, band_col, ["矿化带", "矿化带名称", "zone", "band", "belt"], 0)
            band_series = df[band_real].fillna("").astype(str)
        else:
            band_series = pd.Series(["ALL"] * len(df), index=df.index)

        normalized = pd.DataFrame({
            "band": band_series,
            "x": df[x_real],
            "y": df[y_real],
        })
        return normalized

    if len(df.columns) < 2:
        raise ValueError("无表头模式下，至少需要两列：X、Y")

    if len(df.columns) >= 3 and band_col:
        normalized = df.iloc[:, :3].copy()
        normalized.columns = ["band", "x", "y"]
    else:
        normalized = df.iloc[:, :2].copy()
        normalized.columns = ["x", "y"]
        normalized["band"] = "ALL"
        normalized = normalized[["band", "x", "y"]]
    return normalized


def merge_spatial_independent_gold_points(
    input_file,
    output_file,
    distance_threshold,
    mineralization_band_col,
    x_col,
    y_col,
    log_func=print,
    use_header=True,
):
    """按矿化带和空间距离合并金矿点，输出空间独立样本表。"""

    def log(msg):
        if log_func:
            log_func(msg)

    if distance_threshold is None or distance_threshold <= 0:
        raise ValueError("距离阈值必须为正数")

    df = _read_tabular_input_file(input_file)
    df = _normalize_spatial_input(df, mineralization_band_col, x_col, y_col, use_header=use_header)
    band_col = "band"
    x_col = "x"
    y_col = "y"

    log(f"读取金矿点文件: {input_file}")
    log(f"原始矿点数量: {len(df)}")

    merged_groups = []
    group_counter = 0

    def build_components(points_df):
        coords = points_df[[x_col, y_col]].to_numpy(dtype=float)
        n = len(points_df)
        visited = set()
        components = []

        for start in range(n):
            if start in visited:
                continue
            stack = [start]
            component = []
            visited.add(start)

            while stack:
                i = stack.pop()
                component.append(i)
                for j in range(n):
                    if j in visited:
                        continue
                    dist = float(np.linalg.norm(coords[i] - coords[j]))
                    if dist <= distance_threshold:
                        visited.add(j)
                        stack.append(j)

            components.append(component)
        return components

    for band_value, band_df in df.groupby("band", dropna=False):
        band_df = band_df.reset_index(drop=False)
        log(f"处理矿化带: {band_value}，点数: {len(band_df)}")
        components = build_components(band_df)

        for component in components:
            component_df = band_df.iloc[component].copy()
            group_counter += 1
            center_x = float(component_df[x_col].astype(float).mean())
            center_y = float(component_df[y_col].astype(float).mean())
            merged_groups.append({
                "merged_id": group_counter,
                "mineralization_band": band_value,
                "center_x": center_x,
                "center_y": center_y,
                "point_count": int(len(component_df)),
                "source_indices": ",".join(map(str, component_df["index"].tolist())),
                "source_x": ",".join(map(str, component_df[x_col].tolist())),
                "source_y": ",".join(map(str, component_df[y_col].tolist())),
            })

    result_df = pd.DataFrame(merged_groups)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    _write_tabular_output_file(result_df, output_file)
    log(f"空间独立化完成，输出文件: {output_file}")
    log(f"最终得到 {len(result_df)} 个空间独立的金矿点")


def interpolate_deposit_points(deposit_file, grid_file, output_file, log_func=print):
    """根据矿点坐标标记网格 DAT 文件，生成标签文件"""

    def log(msg):
        if log_func:
            log_func(msg)

    def has_header(file_path, keywords):
        with open(file_path, 'r') as f:
            first_line = f.readline().strip().split()
        return any(token.lower() in keywords for token in first_line)

    log(f"读取矿点文件: {deposit_file}")
    deposit_has_header = has_header(deposit_file, {'x', 'y'})
    deposit_data = pd.read_csv(
        deposit_file,
        sep=r"\s+",
        names=['X', 'Y'] if not deposit_has_header else None,
        header=0 if deposit_has_header else None,
        dtype={'X': float, 'Y': float}
    )

    log(f"读取参考网格文件: {grid_file}")
    grid_has_header = has_header(grid_file, {'x', 'y', 'value'})
    grid_data = pd.read_csv(
        grid_file,
        sep=r"\s+",
        names=['X', 'Y', 'Value'] if not grid_has_header else None,
        header=0 if grid_has_header else None,
        dtype={'X': float, 'Y': float, 'Value': float}
    )

    target_points = grid_data[['X', 'Y']].values
    labels = np.full(len(target_points), -1, dtype=int)

    log(f"开始匹配矿点（共 {len(deposit_data)} 个）…")
    for idx, row in deposit_data.iterrows():
        x, y = row['X'], row['Y']
        distances = np.linalg.norm(target_points - np.array([x, y]), axis=1)
        nearest_point_idx = int(np.argmin(distances))
        labels[nearest_point_idx] = 1
        if (idx + 1) % 100 == 0 or idx == len(deposit_data) - 1:
            log(f"已处理 {idx + 1}/{len(deposit_data)} 个矿点")

    output_df = pd.DataFrame({
        'X': target_points[:, 0],
        'Y': target_points[:, 1],
        'Label': labels
    })

    output_df.to_csv(output_file, sep=' ', index=False, header=False)
    log(f"矿点标签已保存到: {output_file}")


def convert_dat_to_h5(input_file, output_file, grid_shape=(881, 353), log_func=print):
    """将 DAT 网格数据转换为 H5 格式"""

    def log(msg):
        if log_func:
            log_func(msg)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件: {input_file}")

    log(f"读取 DAT 数据: {input_file}")
    data = pd.read_csv(input_file, delimiter=r"\s+", header=None, names=['X', 'Y', 'Value'])

    x_coords = np.sort(data['X'].unique())
    y_coords = np.sort(data['Y'].unique())
    rows, cols = grid_shape

    if len(x_coords) != rows or len(y_coords) != cols:
        log(f"警告：坐标数量 ({len(x_coords)}, {len(y_coords)}) 与期望网格 {grid_shape} 不匹配")

    log("重组网格数据…")
    grid_data = np.zeros((rows, cols), dtype=np.float32)
    x_map = {x: i for i, x in enumerate(x_coords)}
    y_map = {y: j for j, y in enumerate(y_coords)}

    for _, row in data.iterrows():
        i = x_map.get(row['X'])
        j = y_map.get(row['Y'])
        if i is not None and j is not None:
            grid_data[i, j] = row['Value']

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=grid_data)
        f.create_dataset('x_coords', data=x_coords)
        f.create_dataset('y_coords', data=y_coords)

    log(f"转换完成，输出: {output_file}")


def combine_h5_data(input_files, output_file, log_func=print):
    """合并多个 h5 文件的数据，保持 coordinates 不变，拼接 vectors 维度"""
    if not input_files:
        raise ValueError("请至少选择一个输入文件")

    def log(msg):
        if log_func:
            log_func(msg)

    log("开始读取输入文件元数据…")
    with h5py.File(input_files[0], 'r') as f:
        if 'coordinates' in f and 'vectors' in f:
            coordinates = f['coordinates'][:]
            num_points = len(coordinates)
            log("检测到合并格式的文件")
        elif 'x_coords' in f and 'y_coords' in f and 'data' in f:
            x_coords = f['x_coords'][:]
            y_coords = f['y_coords'][:]
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
            coordinates = np.column_stack((X.flatten(), Y.flatten()))
            num_points = len(coordinates)
            log("检测到原始网格格式文件")
        else:
            raise ValueError(f"文件 {input_files[0]} 格式不正确，缺少必要数据集")

    total_dims = 0
    for file_path in input_files:
        with h5py.File(file_path, 'r') as f:
            if 'vectors' in f:
                total_dims += f['vectors'].shape[1]
            elif 'data' in f:
                total_dims += 1
            else:
                raise ValueError(f"文件 {file_path} 缺少 vectors 或 data 数据集")

    log(f"总向量维度：{total_dims}")

    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('coordinates', data=coordinates, dtype=np.float32)
        vectors = f_out.create_dataset('vectors', shape=(num_points, total_dims), dtype=np.float32)

        current_dim = 0
        for idx, file_path in enumerate(input_files, 1):
            log(f"处理文件 {idx}/{len(input_files)}: {file_path}")
            with h5py.File(file_path, 'r') as f_in:
                if 'coordinates' in f_in:
                    file_vectors = f_in['vectors'][:]
                    if file_vectors.shape[0] != num_points:
                        raise ValueError(f"文件 {file_path} 的数据点数量与基准文件不一致")
                    if not np.allclose(f_in['coordinates'][:], coordinates):
                        log(f"警告：文件 {file_path} 的坐标与基准文件不完全一致")
                    dims = file_vectors.shape[1]
                else:
                    x_in = f_in['x_coords'][:]
                    y_in = f_in['y_coords'][:]
                    X_in, Y_in = np.meshgrid(x_in, y_in, indexing='ij')
                    coords_in = np.column_stack((X_in.flatten(), Y_in.flatten()))
                    if not np.allclose(coords_in, coordinates):
                        log(f"警告：文件 {file_path} 的坐标与基准文件不完全一致")
                    file_vectors = f_in['data'][:].flatten()[:, np.newaxis]
                    dims = 1

                vectors[:, current_dim:current_dim + dims] = file_vectors
                current_dim += dims
                log(f"已写入维度范围 {current_dim - dims} - {current_dim - 1}")

        f_out.attrs['file_names'] = [os.path.basename(p) for p in input_files]

    log(f"合并完成，输出文件：{output_file}")


def load_magnetic_data(filename, log_func=print):
    def log(msg):
        if log_func:
            log_func(msg)

    with h5py.File(filename, 'r') as f:
        coordinates = f['coordinates'][:]  # (N, 2)
        vectors = f['vectors'][:]

    log(f"coordinates 形状: {coordinates.shape}")
    log(f"vectors 形状: {vectors.shape}")

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    x_grid, y_grid = np.meshgrid(unique_x, unique_y, indexing='ij')

    n_features = vectors.shape[1]
    magnetic_grid = np.zeros((len(unique_x), len(unique_y), n_features), dtype=np.float32)
    for i in range(n_features):
        magnetic_grid[:, :, i] = griddata((x, y), vectors[:, i], (x_grid, y_grid), method='cubic')

    log(f"网格化数据形状: {magnetic_grid.shape}")
    return x_grid, y_grid, magnetic_grid


def pad_data(data, window_size, pad_mode='edge', log_func=print):
    def log(msg):
        if log_func:
            log_func(msg)

    pad_size = window_size // 2
    if pad_size == 0:
        return data

    log(f"按 {pad_mode} 模式扩边，pad_size={pad_size}")
    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    if pad_mode == 'constant':
        padded = np.pad(data, pad_width, mode='constant', constant_values=0)
    else:
        padded = np.pad(data, pad_width, mode=pad_mode)

    log(f"扩边后数据形状: {padded.shape}")
    return padded


def sliding_window(data, window_size=32, stride=2, log_func=print):
    def log(msg):
        if log_func:
            log_func(msg)

    windows = []
    positions = []
    rows, cols, n_features = data.shape
    n_rows = rows - window_size + 1
    n_cols = cols - window_size + 1

    log(f"开始滑动窗口切割，窗口大小 {window_size}，步长 {stride}")

    for i in range(0, n_rows, stride):
        for j in range(0, n_cols, stride):
            window = data[i:i + window_size, j:j + window_size, :]
            if window.shape == (window_size, window_size, n_features):
                windows.append(window)
                positions.append((i + window_size // 2, j + window_size // 2))

    windows = np.array(windows)
    windows = np.transpose(windows, (1, 2, 3, 0))  # (window, window, features, N)
    positions = np.array(positions)

    log(f"共生成 {windows.shape[3]} 个窗口")
    return windows, positions


def index_to_geo_coords(positions, x_offset, y_offset, x_max, y_max, rows, cols):
    geo_positions = np.zeros_like(positions, dtype=float)
    x_scale = (x_max - x_offset) / (rows - 1) if rows > 1 else 1
    y_scale = (y_max - y_offset) / (cols - 1) if cols > 1 else 1
    geo_positions[:, 0] = x_offset + positions[:, 0] * x_scale
    geo_positions[:, 1] = y_offset + positions[:, 1] * y_scale
    return geo_positions


def save_windows_to_h5(filename, windows, positions, window_size, x_offset, y_offset, orig_shape,
                        x_max, y_max, log_func=print):
    def log(msg):
        if log_func:
            log_func(msg)

    rows, cols = orig_shape[:2]
    log(f"保存窗口到 {filename}")
    log(f"原始数据形状: {rows}x{cols}")
    geo_positions = index_to_geo_coords(positions, x_offset, y_offset, x_max, y_max, rows, cols)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('windows', data=windows)
        f.create_dataset('positions', data=geo_positions)
        f.create_dataset('index_positions', data=positions)
        f.attrs['window_size'] = window_size
        f.attrs['x_offset'] = x_offset
        f.attrs['y_offset'] = y_offset
        f.attrs['x_scale'] = (x_max - x_offset) / (rows - 1) if rows > 1 else 1
        f.attrs['y_scale'] = (y_max - y_offset) / (cols - 1) if cols > 1 else 1
        f.attrs['x_max'] = x_max
        f.attrs['y_max'] = y_max
        f.attrs['data_rows'] = rows
        f.attrs['data_cols'] = cols

    log("窗口数据保存完成")

class DataProcessingWindow(QMainWindow):
    run_on_main = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据合并与切割工具")
        self.resize(1100, 780)
        self.merge_files: list[str] = []
        self.grid_shape = {"rows": None, "cols": None}

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._flush_log_queue)
        self.log_timer.start(150)

        self.run_on_main.connect(self._execute_main_callback)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_deposit_tab()
        self._build_grd_fusion_tab()
        self._build_convert_tab()
        self._build_merge_tab()
        self._build_slice_tab()
        self._build_spatial_dedup_tab()

        layout.addWidget(self._build_log_panel(), 1)

    # --- UI builders -----------------------------------------------------
    def _build_deposit_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(10)

        info = QLabel("输入矿点 DAT 与参考 Ag.dat，生成含 Label 列的数据文件")
        info.setStyleSheet("color: #555555;")
        tab_layout.addWidget(info)

        file_group = QGroupBox("文件设置")
        form = QFormLayout(file_group)
        form.setLabelAlignment(Qt.AlignLeft)

        self.deposit_input_edit = QLineEdit()
        form.addRow("矿点文件 (deposit):", self._file_input_widget(self.deposit_input_edit, self.select_deposit_input))

        self.grid_input_edit = QLineEdit()
        form.addRow("参考网格 (dat):", self._file_input_widget(self.grid_input_edit, self.select_grid_input))

        self.deposit_output_edit = QLineEdit()
        form.addRow("输出文件:", self._file_input_widget(self.deposit_output_edit, self.select_deposit_output))

        tab_layout.addWidget(file_group)

        self.deposit_run_btn = QPushButton("生成矿点标签")
        self.deposit_run_btn.clicked.connect(self.run_deposit_interpolation)
        tab_layout.addWidget(self.deposit_run_btn, alignment=Qt.AlignLeft)

        tab_layout.addStretch(1)
        self.tabs.addTab(tab, "矿点标签")

    def _build_grd_fusion_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(10)

        info = QLabel("将多个 GRD 文件按顺序融合为单个 H5 文件，适用于后续切割与模型训练。")
        info.setStyleSheet("color: #555555;")
        info.setWordWrap(True)
        tab_layout.addWidget(info)

        file_group = QGroupBox("文件设置")
        form = QFormLayout(file_group)

        self.grd_dir_edit = QLineEdit()
        self.grd_dir_edit.setReadOnly(True)
        form.addRow("GRD 目录:", self._file_input_widget(self.grd_dir_edit, self.select_grd_directory))

        self.grd_available_list = QListWidget()
        self.grd_available_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        form.addRow("可用 GRD 文件:", self.grd_available_list)

        self.grd_selected_list = QListWidget()
        self.grd_selected_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grd_selected_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.grd_selected_list.setDefaultDropAction(Qt.MoveAction)
        form.addRow("待融合文件:", self.grd_selected_list)

        tab_layout.addWidget(file_group)

        buttons = QHBoxLayout()
        self.grd_add_btn = QPushButton("添加选中")
        self.grd_add_btn.clicked.connect(self.add_selected_grd)
        self.grd_add_all_btn = QPushButton("添加全部")
        self.grd_add_all_btn.clicked.connect(self.add_all_grd)
        self.grd_remove_btn = QPushButton("移除选中")
        self.grd_remove_btn.clicked.connect(self.remove_selected_grd)
        self.grd_clear_btn = QPushButton("清空")
        self.grd_clear_btn.clicked.connect(self.clear_grd_selection)
        for btn in (self.grd_add_btn, self.grd_add_all_btn, self.grd_remove_btn, self.grd_clear_btn):
            buttons.addWidget(btn)
        buttons.addStretch(1)
        tab_layout.addLayout(buttons)

        output_group = QGroupBox("输出设置")
        output_form = QFormLayout(output_group)
        self.grd_output_edit = QLineEdit()
        output_form.addRow("输出 H5 文件:", self._file_input_widget(self.grd_output_edit, self.select_grd_output))
        tab_layout.addWidget(output_group)

        preprocess_group = QGroupBox("预处理设置")
        preprocess_form = QFormLayout(preprocess_group)
        self.grd_norm_combo = QComboBox()
        self.grd_norm_combo.addItems(["不处理", "Z-score 标准化", "0-1 标准化"])
        preprocess_form.addRow("归一化方式:", self.grd_norm_combo)
        self.grd_norm_params_edit = QLineEdit()
        preprocess_form.addRow(
            "参数 JSON(可选):",
            self._file_input_widget(self.grd_norm_params_edit, self.select_grd_norm_params_file),
        )
        tab_layout.addWidget(preprocess_group)

        self.grd_fusion_run_btn = QPushButton("开始融合")
        self.grd_fusion_run_btn.clicked.connect(self.run_grd_fusion)
        tab_layout.addWidget(self.grd_fusion_run_btn, alignment=Qt.AlignLeft)

        tab_layout.addStretch(1)
        self.tabs.addTab(tab, "GRD 数据融合")

    def _build_convert_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(10)

        info = QLabel("将 DAT 网格数据转换为 H5 文件，并自动识别行列网格尺寸。")
        info.setStyleSheet("color: #555555;")
        info.setWordWrap(True)
        tab_layout.addWidget(info)

        file_group = QGroupBox("文件设置")
        form = QFormLayout(file_group)

        self.dat_input_edit = QLineEdit()
        form.addRow("DAT 输入:", self._file_input_widget(self.dat_input_edit, self.select_dat_input))

        self.dat_output_edit = QLineEdit()
        form.addRow("H5 输出:", self._file_input_widget(self.dat_output_edit, self.select_dat_output))

        tab_layout.addWidget(file_group)

        grid_group = QGroupBox("网格参数")
        grid_layout = QGridLayout(grid_group)
        grid_layout.addWidget(QLabel("行数 (X)"), 0, 0)
        self.grid_rows_display = QLineEdit()
        self.grid_rows_display.setReadOnly(True)
        self.grid_rows_display.setPlaceholderText("未识别")
        grid_layout.addWidget(self.grid_rows_display, 0, 1)
        grid_layout.addWidget(QLabel("列数 (Y)"), 0, 2)
        self.grid_cols_display = QLineEdit()
        self.grid_cols_display.setReadOnly(True)
        self.grid_cols_display.setPlaceholderText("未识别")
        grid_layout.addWidget(self.grid_cols_display, 0, 3)

        tab_layout.addWidget(grid_group)

        self.convert_run_btn = QPushButton("执行转换")
        self.convert_run_btn.clicked.connect(self.run_dat_convert)
        tab_layout.addWidget(self.convert_run_btn, alignment=Qt.AlignLeft)

        tab_layout.addStretch(1)
        self.tabs.addTab(tab, "DAT 转 H5")

    def _build_merge_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(10)

        input_group = QGroupBox("输入文件")
        input_layout = QVBoxLayout(input_group)
        buttons_row = QHBoxLayout()
        select_btn = QPushButton("选择 h5 文件")
        select_btn.clicked.connect(self.select_merge_input)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_merge_input)
        buttons_row.addWidget(select_btn)
        buttons_row.addWidget(clear_btn)
        buttons_row.addStretch(1)
        input_layout.addLayout(buttons_row)

        self.merge_list_widget = QListWidget()
        input_layout.addWidget(self.merge_list_widget)
        tab_layout.addWidget(input_group)

        output_group = QGroupBox("输出设置")
        output_form = QFormLayout(output_group)
        self.merge_output_edit = QLineEdit()
        output_form.addRow("输出文件:", self._file_input_widget(self.merge_output_edit, self.select_merge_output))
        tab_layout.addWidget(output_group)

        self.merge_run_btn = QPushButton("执行合并")
        self.merge_run_btn.clicked.connect(self.run_merge)
        tab_layout.addWidget(self.merge_run_btn, alignment=Qt.AlignLeft)

        tab_layout.addStretch(1)
        self.tabs.addTab(tab, "数据合并")

    def _build_slice_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(10)

        file_group = QGroupBox("文件设置")
        file_form = QFormLayout(file_group)
        self.slice_input_edit = QLineEdit()
        file_form.addRow("输入文件:", self._file_input_widget(self.slice_input_edit, self.select_slice_input))
        self.slice_output_edit = QLineEdit()
        file_form.addRow("输出文件:", self._file_input_widget(self.slice_output_edit, self.select_slice_output))
        tab_layout.addWidget(file_group)

        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout(param_group)
        self.window_size_edit = QLineEdit()
        self.stride_edit = QLineEdit()
        self.pad_mode_combo = QComboBox()
        self.pad_mode_combo.addItems([label for label, _ in PAD_MODE_OPTIONS])
        self.x_min_display = QLineEdit()
        self.x_max_display = QLineEdit()
        self.y_min_display = QLineEdit()
        self.y_max_display = QLineEdit()
        for widget in (self.x_min_display, self.x_max_display, self.y_min_display, self.y_max_display):
            widget.setReadOnly(True)

        labels = [
            ("窗口大小", self.window_size_edit),
            ("步长", self.stride_edit),
            ("扩边模式", self.pad_mode_combo),
            ("X 最小", self.x_min_display),
            ("X 最大", self.x_max_display),
            ("Y 最小", self.y_min_display),
            ("Y 最大", self.y_max_display),
        ]

        for row, (text, widget) in enumerate(labels):
            param_layout.addWidget(QLabel(text), row // 2, (row % 2) * 2)
            param_layout.addWidget(widget, row // 2, (row % 2) * 2 + 1)

        tab_layout.addWidget(param_group)

        self.slice_run_btn = QPushButton("执行切割")
        self.slice_run_btn.clicked.connect(self.run_slice)
        tab_layout.addWidget(self.slice_run_btn, alignment=Qt.AlignLeft)

        tab_layout.addStretch(1)
        self.tabs.addTab(tab, "数据切割")

    def select_merge_input(self):
        files = self._select_files("选择需要合并的 h5 文件", "H5 文件 (*.h5);;所有文件 (*.*)")
        if files:
            self.merge_files = files
            self.merge_list_widget.clear()
            self.merge_list_widget.addItems(files)

    def clear_merge_input(self):
        self.merge_files = []
        self.merge_list_widget.clear()

    def select_merge_output(self):
        path = self._select_save_path("选择合并输出文件", "H5 文件 (*.h5);;所有文件 (*.*)", ".h5")
        if path:
            self.merge_output_edit.setText(path)

    def run_merge(self):
        if not self.merge_files:
            QMessageBox.warning(self, "提示", "请先选择至少一个输入文件")
            return

        output_path = self.merge_output_edit.text().strip()
        if not output_path:
            self.select_merge_output()
            output_path = self.merge_output_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "提示", "请指定输出文件路径")
            return

        self.merge_run_btn.setEnabled(False)
        self.log("开始执行数据合并…")

        def task():
            try:
                combine_h5_data(self.merge_files, output_path, self.log)
                self._show_message("合并完成", f"输出文件：{output_path}")
            except Exception as exc:
                self.log(f"合并失败：{exc}")
                self._show_message("错误", str(exc), error=True)
            finally:
                self._call_on_main(lambda: self.merge_run_btn.setEnabled(True))

        threading.Thread(target=task, daemon=True).start()

    def _build_spatial_dedup_tab(self):
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(10)

        info = QLabel("对已知金矿点按“同一矿化带 + 距离阈值”进行空间去重合并，生成独立样本。")
        info.setStyleSheet("color: #555555;")
        info.setWordWrap(True)
        tab_layout.addWidget(info)

        file_group = QGroupBox("文件设置")
        file_form = QFormLayout(file_group)
        self.spatial_input_edit = QLineEdit()
        file_form.addRow("输入金矿点文件:", self._file_input_widget(self.spatial_input_edit, self.select_spatial_input))
        self.spatial_output_edit = QLineEdit()
        file_form.addRow("输出结果文件:", self._file_input_widget(self.spatial_output_edit, self.select_spatial_output))
        tab_layout.addWidget(file_group)

        param_group = QGroupBox("参数设置")
        param_form = QFormLayout(param_group)
        self.distance_threshold_edit = QLineEdit()
        self.distance_threshold_edit.setPlaceholderText("例如：500，单位与坐标一致")
        self.header_mode_combo = QComboBox()
        self.header_mode_combo.addItems(["自动识别表头", "无表头（前2列依次为X/Y 或前3列为矿化带/X/Y）"])
        self.band_col_combo = QComboBox()
        self.x_col_combo = QComboBox()
        self.y_col_combo = QComboBox()
        self.band_col_combo.setEditable(True)
        self.x_col_combo.setEditable(True)
        self.y_col_combo.setEditable(True)
        self.band_col_combo.addItems(["矿化带", "矿带", "成矿带", "belt", "zone"])
        self.x_col_combo.addItems(["X", "x", "lon", "longitude", "经度"])
        self.y_col_combo.addItems(["Y", "y", "lat", "latitude", "纬度"])
        param_form.addRow("距离阈值:", self.distance_threshold_edit)
        param_form.addRow("数据格式:", self.header_mode_combo)
        param_form.addRow("矿化带字段:", self.band_col_combo)
        param_form.addRow("X 坐标字段:", self.x_col_combo)
        param_form.addRow("Y 坐标字段:", self.y_col_combo)
        tab_layout.addWidget(param_group)

        self.spatial_run_btn = QPushButton("执行空间独立化")
        self.spatial_run_btn.clicked.connect(self.run_spatial_dedup)
        tab_layout.addWidget(self.spatial_run_btn, alignment=Qt.AlignLeft)

        tab_layout.addStretch(1)
        self.tabs.addTab(tab, "空间独立化")

    def _build_log_panel(self) -> QGroupBox:
        group = QGroupBox("运行日志")
        layout = QVBoxLayout(group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        return group

    # --- Helpers --------------------------------------------------------
    def _file_input_widget(self, line_edit: QLineEdit, handler, button_label: str = "选择…") -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(line_edit)
        button = QPushButton(button_label)
        button.clicked.connect(handler)
        row.addWidget(button)
        return container

    def _execute_main_callback(self, func):
        if callable(func):
            func()

    def _call_on_main(self, func):
        """Schedule *func* to run on the main GUI thread."""
        if threading.current_thread() is threading.main_thread():
            func()
        else:
            self.run_on_main.emit(func)

    def log(self, message: str):
        timestamp = time.strftime('%H:%M:%S')
        self.log_queue.put(f"[{timestamp}] {message}")

    def _flush_log_queue(self):
        if self.log_queue.empty():
            return
        while not self.log_queue.empty():
            message = self.log_queue.get()
            self.log_view.appendPlainText(message)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _select_file(self, title: str, filter_str: str) -> str:
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        return path

    def _scan_spatial_input_columns(self, path: str):
        try:
            df = _read_tabular_input_file(path)
        except Exception as exc:
            self.log(f"无法读取表头: {exc}")
            return

        columns = [str(col) for col in df.columns.tolist()]
        band_candidates = ["矿化带", "矿带", "成矿带", "belt", "zone"]
        x_candidates = ["X", "x", "lon", "longitude", "经度"]
        y_candidates = ["Y", "y", "lat", "latitude", "纬度"]

        def fill_combo(combo, candidates):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            for candidate in candidates:
                if candidate in columns:
                    combo.setCurrentText(candidate)
                    break
            combo.blockSignals(False)

        fill_combo(self.band_col_combo, band_candidates)
        fill_combo(self.x_col_combo, x_candidates)
        fill_combo(self.y_col_combo, y_candidates)
        self.log(f"已识别字段: {', '.join(columns)}")

    def _select_files(self, title: str, filter_str: str) -> list[str]:
        paths, _ = QFileDialog.getOpenFileNames(self, title, "", filter_str)
        return paths

    def _select_save_path(self, title: str, filter_str: str, default_suffix: str) -> str:
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            title,
            "",
            filter_str,
            options=QFileDialog.DontConfirmOverwrite,
        )
        if path and not os.path.splitext(path)[1]:
            ext_map = {
                "CSV 文件 (*.csv)": ".csv",
                "Excel 文件 (*.xlsx)": ".xlsx",
                "Excel 97-2003 文件 (*.xls)": ".xls",
                "DAT 文件 (*.dat)": ".dat",
                "TXT 文件 (*.txt)": ".txt",
            }
            path += ext_map.get(selected_filter, default_suffix)
        return path

    def _show_message(self, title: str, message: str, error: bool = False):
        def _show():
            if error:
                QMessageBox.critical(self, title, message)
            else:
                QMessageBox.information(self, title, message)

        self._call_on_main(_show)

    # --- Deposit actions ------------------------------------------------
    def select_deposit_input(self):
        path = self._select_file("选择矿点 deposit 文件", "DAT 文件 (*.dat);;所有文件 (*.*)")
        if path:
            self.deposit_input_edit.setText(path)
            if not self.deposit_output_edit.text():
                base = os.path.splitext(os.path.basename(path))[0]
                suggested = os.path.join(os.path.dirname(path), f"{base}_label.dat")
                self.deposit_output_edit.setText(suggested)

    def select_grid_input(self):
        path = self._select_file("选择网格 dat 文件", "DAT 文件 (*.dat);;所有文件 (*.*)")
        if path:
            self.grid_input_edit.setText(path)

    def select_deposit_output(self):
        path = self._select_save_path("选择输出标签文件", "DAT 文件 (*.dat);;所有文件 (*.*)", ".dat")
        if path:
            self.deposit_output_edit.setText(path)

    def run_deposit_interpolation(self):
        deposit_file = self.deposit_input_edit.text().strip()
        grid_file = self.grid_input_edit.text().strip()
        output_file = self.deposit_output_edit.text().strip()

        if not deposit_file:
            QMessageBox.warning(self, "提示", "请先选择矿点 deposit 文件")
            return
        if not os.path.exists(deposit_file):
            QMessageBox.critical(self, "错误", "矿点文件不存在")
            return
        if not grid_file:
            QMessageBox.warning(self, "提示", "请先选择网格 dat 文件")
            return
        if not os.path.exists(grid_file):
            QMessageBox.critical(self, "错误", "网格 dat 文件不存在")
            return
        if not output_file:
            self.select_deposit_output()
            output_file = self.deposit_output_edit.text().strip()
        if not output_file:
            QMessageBox.warning(self, "提示", "请指定输出文件")
            return

        self.deposit_run_btn.setEnabled(False)
        self.log("开始生成矿点标签…")

        def task():
            try:
                interpolate_deposit_points(deposit_file, grid_file, output_file, self.log)
                self._show_message("生成完成", f"输出文件：{output_file}")
            except Exception as exc:
                self.log(f"生成矿点标签失败：{exc}")
                self._show_message("错误", str(exc), error=True)
            finally:
                self._call_on_main(lambda: self.deposit_run_btn.setEnabled(True))

        threading.Thread(target=task, daemon=True).start()

    # --- GRD fusion actions --------------------------------------------
    def select_grd_directory(self):
        path = QFileDialog.getExistingDirectory(self, "选择 GRD 文件夹", self.grd_dir_edit.text().strip() or os.getcwd())
        if path:
            self.grd_dir_edit.setText(path)
            self._populate_grd_files(path)

    def _populate_grd_files(self, directory):
        self.grd_available_list.clear()
        grd_files = []
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith('.grd'):
                grd_files.append(os.path.join(directory, name))
        for path in grd_files:
            self.grd_available_list.addItem(path)
        self.log(f"已加载 {len(grd_files)} 个 GRD 文件")

    def add_selected_grd(self):
        items = self.grd_available_list.selectedItems()
        for item in items:
            path = item.text()
            if not self._list_contains(self.grd_selected_list, path):
                self.grd_selected_list.addItem(path)

    def add_all_grd(self):
        for index in range(self.grd_available_list.count()):
            path = self.grd_available_list.item(index).text()
            if not self._list_contains(self.grd_selected_list, path):
                self.grd_selected_list.addItem(path)

    def remove_selected_grd(self):
        for item in self.grd_selected_list.selectedItems():
            self.grd_selected_list.takeItem(self.grd_selected_list.row(item))

    def clear_grd_selection(self):
        self.grd_selected_list.clear()

    def select_grd_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "选择输出 H5 文件", "", "H5 文件 (*.h5);;所有文件 (*.*)")
        if path:
            self.grd_output_edit.setText(path)

    def select_grd_norm_params_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标准化参数文件",
            "",
            "JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if path:
            self.grd_norm_params_edit.setText(path)

    def run_grd_fusion(self):
        input_files = [self.grd_selected_list.item(i).text() for i in range(self.grd_selected_list.count())]
        output_file = self.grd_output_edit.text().strip()
        norm_map = {
            "不处理": "none",
            "Z-score 标准化": "zscore",
            "0-1 标准化": "normalize",
        }
        normalize_method = norm_map.get(self.grd_norm_combo.currentText(), "none")
        params_path = self.grd_norm_params_edit.text().strip() or None
        if params_path and normalize_method == "none":
            reply = QMessageBox.question(
                self,
                "确认操作",
                "你选择了参数文件，但归一化方式为“不处理”。参数文件将被忽略，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return
        if not input_files:
            QMessageBox.warning(self, "提示", "请至少选择一个 GRD 文件")
            return
        if not output_file:
            self.select_grd_output()
            output_file = self.grd_output_edit.text().strip()
        if not output_file:
            QMessageBox.warning(self, "提示", "请指定输出文件")
            return

        self.grd_fusion_run_btn.setEnabled(False)
        self.log("开始融合 GRD 文件…")

        def task():
            try:
                combine_grid_files_to_h5(
                    input_files,
                    output_file,
                    normalize_method=normalize_method,
                    params_path=params_path,
                    log_func=self.log,
                )
                self._show_message("融合完成", f"输出文件：{output_file}")
            except Exception as exc:
                self.log(f"GRD 融合失败：{exc}")
                self._show_message("错误", str(exc), error=True)
            finally:
                self._call_on_main(lambda: self.grd_fusion_run_btn.setEnabled(True))

        threading.Thread(target=task, daemon=True).start()

    def _list_contains(self, list_widget, text):
        for idx in range(list_widget.count()):
            if list_widget.item(idx).text() == text:
                return True
        return False

    # --- Convert actions ------------------------------------------------
    def select_dat_input(self):
        path = self._select_file("选择 DAT 文件", "DAT 文件 (*.dat);;所有文件 (*.*)")
        if path:
            self.dat_input_edit.setText(path)
            if not self.dat_output_edit.text():
                self.dat_output_edit.setText(os.path.splitext(path)[0] + '.h5')
            self.grid_shape = {"rows": None, "cols": None}
            self.grid_rows_display.setText("识别中…")
            self.grid_cols_display.setText("识别中…")
            self._fill_grid_shape_from_dat(path)

    def select_dat_output(self):
        path = self._select_save_path("选择输出 H5 文件", "H5 文件 (*.h5);;所有文件 (*.*)", ".h5")
        if path:
            self.dat_output_edit.setText(path)

    def run_dat_convert(self):
        dat_path = self.dat_input_edit.text().strip()
        output_path = self.dat_output_edit.text().strip()

        if not dat_path:
            QMessageBox.warning(self, "提示", "请先选择 DAT 输入文件")
            return
        if not os.path.exists(dat_path):
            QMessageBox.critical(self, "错误", "DAT 输入文件不存在")
            return
        if not output_path:
            self.select_dat_output()
            output_path = self.dat_output_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "提示", "请指定输出 H5 文件")
            return

        rows = self.grid_shape.get("rows")
        cols = self.grid_shape.get("cols")
        if not rows or not cols:
            QMessageBox.critical(self, "错误", "请先识别网格行/列数")
            return

        self.convert_run_btn.setEnabled(False)
        self.log("开始进行 DAT→H5 转换…")

        def task():
            try:
                convert_dat_to_h5(dat_path, output_path, (rows, cols), self.log)
                self._show_message("转换完成", f"输出文件：{output_path}")
            except Exception as exc:
                self.log(f"转换失败：{exc}")
                self._show_message("错误", str(exc), error=True)
            finally:
                self._call_on_main(lambda: self.convert_run_btn.setEnabled(True))

        threading.Thread(target=task, daemon=True).start()

    def _fill_grid_shape_from_dat(self, dat_path: str):
        def task():
            try:
                data = pd.read_csv(dat_path, delimiter=r"\s+", header=None, names=['X', 'Y', 'Value'])
                x_count = int(len(np.unique(data['X'])))
                y_count = int(len(np.unique(data['Y'])))

                def update():
                    self.grid_shape['rows'] = x_count
                    self.grid_shape['cols'] = y_count
                    self.grid_rows_display.setText(str(x_count))
                    self.grid_cols_display.setText(str(y_count))

                self._call_on_main(update)
                self.log(f"自动识别网格参数: 行={x_count}, 列={y_count}")
            except Exception as exc:
                self.log(f"无法识别网格参数: {exc}")
                self._call_on_main(lambda: (self.grid_rows_display.setText("识别失败"), self.grid_cols_display.setText("识别失败")))

        threading.Thread(target=task, daemon=True).start()

    # --- Spatial dedup actions -----------------------------------------
    def select_spatial_input(self):
        path = self._select_file(
            "选择金矿点文件",
            "表格文件 (*.csv *.xls *.xlsx *.dat *.txt);;CSV 文件 (*.csv);;Excel 文件 (*.xls *.xlsx);;DAT/TXT 文件 (*.dat *.txt);;所有文件 (*.*)",
        )
        if path:
            self.spatial_input_edit.setText(path)
            self._scan_spatial_input_columns(path)
            if not self.spatial_output_edit.text():
                base = os.path.splitext(os.path.basename(path))[0]
                suggested = os.path.join(os.path.dirname(path), f"{base}_spatial_independent.csv")
                self.spatial_output_edit.setText(suggested)

    def select_spatial_output(self):
        path = self._select_save_path(
            "选择输出文件",
            "CSV 文件 (*.csv);;Excel 文件 (*.xlsx);;Excel 97-2003 文件 (*.xls);;DAT 文件 (*.dat);;TXT 文件 (*.txt);;所有文件 (*.*)",
            ".csv",
        )
        if path:
            self.spatial_output_edit.setText(path)

    def run_spatial_dedup(self):
        input_path = self.spatial_input_edit.text().strip()
        output_path = self.spatial_output_edit.text().strip()
        threshold_text = self.distance_threshold_edit.text().strip()
        band_col = self.band_col_combo.currentText().strip()
        x_col = self.x_col_combo.currentText().strip()
        y_col = self.y_col_combo.currentText().strip()
        use_header = self.header_mode_combo.currentIndex() == 0
        if not use_header:
            band_col = ""

        if not input_path:
            QMessageBox.warning(self, "提示", "请先选择金矿点文件")
            return
        if not os.path.exists(input_path):
            QMessageBox.critical(self, "错误", "金矿点输入文件不存在")
            return
        if not output_path:
            self.select_spatial_output()
            output_path = self.spatial_output_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "提示", "请指定输出文件路径")
            return
        if not threshold_text:
            QMessageBox.warning(self, "提示", "请填写距离阈值")
            return

        try:
            threshold = float(threshold_text)
        except ValueError:
            QMessageBox.critical(self, "错误", "距离阈值必须是数值")
            return

        if not x_col or not y_col:
            QMessageBox.warning(self, "提示", "请至少填写 X/Y 坐标字段名")
            return
        if use_header and band_col == "":
            band_col = ""

        self.spatial_run_btn.setEnabled(False)
        self.log("开始执行金矿点空间独立化…")

        def task():
            try:
                merge_spatial_independent_gold_points(
                    input_path,
                    output_path,
                    threshold,
                    band_col,
                    x_col,
                    y_col,
                    self.log,
                    use_header=use_header,
                )
                self._show_message("处理完成", f"输出文件：{output_path}")
            except Exception as exc:
                self.log(f"空间独立化失败：{exc}")
                self._show_message("错误", str(exc), error=True)
            finally:
                self._call_on_main(lambda: self.spatial_run_btn.setEnabled(True))

        threading.Thread(target=task, daemon=True).start()

    # --- Slice actions --------------------------------------------------
    def select_slice_input(self):
        path = self._select_file("选择需要切割的 h5 文件", "H5 文件 (*.h5);;所有文件 (*.*)")
        if path:
            self.slice_input_edit.setText(path)
            if not self.slice_output_edit.text():
                directory, name = os.path.split(path)
                default_name = f"windows_{os.path.splitext(name)[0]}.h5"
                self.slice_output_edit.setText(os.path.join(directory, default_name))
            self._fill_slice_coord_bounds(path)

    def select_slice_output(self):
        path = self._select_save_path("选择切割输出文件", "H5 文件 (*.h5);;所有文件 (*.*)", ".h5")
        if path:
            self.slice_output_edit.setText(path)

    def run_slice(self):
        input_path = self.slice_input_edit.text().strip()
        output_path = self.slice_output_edit.text().strip()

        if not input_path:
            QMessageBox.warning(self, "提示", "请先选择输入文件")
            return
        if not os.path.exists(input_path):
            QMessageBox.critical(self, "错误", "输入文件不存在")
            return
        if not output_path:
            self.select_slice_output()
            output_path = self.slice_output_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "提示", "请指定输出文件路径")
            return

        try:
            window_size = int(self.window_size_edit.text())
            stride = int(self.stride_edit.text())
            x_min = float(self.x_min_display.text())
            x_max = float(self.x_max_display.text())
            y_min = float(self.y_min_display.text())
            y_max = float(self.y_max_display.text())
        except ValueError:
            QMessageBox.critical(self, "错误", "请确保窗口、步长和坐标范围输入有效")
            return

        pad_mode_label = self.pad_mode_combo.currentText()
        pad_mode = PAD_LABEL_TO_VALUE.get(pad_mode_label, PAD_MODE_OPTIONS[0][1])
        if window_size <= 0 or stride <= 0:
            QMessageBox.critical(self, "错误", "窗口大小和步长必须为正数")
            return
        if x_max <= x_min or y_max <= y_min:
            QMessageBox.critical(self, "错误", "请确认坐标范围（最大值需大于最小值）")
            return

        self.slice_run_btn.setEnabled(False)
        self.log("开始执行数据切割…")

        def task():
            try:
                _, _, magnetic_grid = load_magnetic_data(input_path, self.log)
                rows, cols = magnetic_grid.shape[:2]
                self.log(f"数据维度: {rows}×{cols}")

                padded_data = pad_data(magnetic_grid, window_size, pad_mode, self.log)
                windows, positions = sliding_window(padded_data, window_size, stride, self.log)
                positions = positions - window_size // 2

                save_windows_to_h5(output_path, windows, positions, window_size,
                                   x_min, y_min, magnetic_grid.shape, x_max, y_max, self.log)

                self._show_message("切割完成", f"输出文件：{output_path}")
            except Exception as exc:
                self.log(f"切割失败：{exc}")
                self._show_message("错误", str(exc), error=True)
            finally:
                self._call_on_main(lambda: self.slice_run_btn.setEnabled(True))

        threading.Thread(target=task, daemon=True).start()

    def _fill_slice_coord_bounds(self, h5_path: str):
        def task():
            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'coordinates' in f:
                        coords = f['coordinates'][:]
                        x_vals = coords[:, 0]
                        y_vals = coords[:, 1]
                    elif 'x_coords' in f and 'y_coords' in f:
                        x_vals = f['x_coords'][:]
                        y_vals = f['y_coords'][:]
                    else:
                        raise ValueError('未找到坐标数据集')

                x_min = float(np.min(x_vals))
                x_max = float(np.max(x_vals))
                y_min = float(np.min(y_vals))
                y_max = float(np.max(y_vals))

                def update_fields():
                    self.x_min_display.setText(f"{x_min:.6f}")
                    self.x_max_display.setText(f"{x_max:.6f}")
                    self.y_min_display.setText(f"{y_min:.6f}")
                    self.y_max_display.setText(f"{y_max:.6f}")

                self._call_on_main(update_fields)
                self.log(f"自动识别坐标范围: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
            except Exception as exc:
                self.log(f"无法识别坐标范围: {exc}")
                self._call_on_main(lambda: (self.x_min_display.clear(), self.x_max_display.clear(),
                                            self.y_min_display.clear(), self.y_max_display.clear()))

        threading.Thread(target=task, daemon=True).start()


def main():
    import sys

    app = QApplication(sys.argv)
    window = DataProcessingWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
