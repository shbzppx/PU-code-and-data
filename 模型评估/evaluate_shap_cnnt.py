"""
CNNTransformer 模型 SHAP 可解释性分析工具

📋 快速配置指南：
1. 在 main() 函数中找到 "手动配置区域"
2. 修改对应的参数值
3. 保存文件后运行: python evaluate_shap_cnnt.py

🔧 主要参数说明：

文件路径参数：
- MODEL_PATH: 训练好的模型文件路径
- INPUT_DATA: 输入数据文件路径  
- LABEL_FILE: 标签文件路径

核心分析参数：
- NUM_BACKGROUND_SAMPLES: SHAP背景样本数 (20-200, 越大越准确但越慢)
- NUM_TEST_SAMPLES: 测试样本数 (50-1000, 越大分析越全面)
- DEVICE: 'cuda' 或 'cpu'


💡 常用配置示例：
# 快速测试 (节省时间和内存):
# NUM_BACKGROUND_SAMPLES = 20, NUM_TEST_SAMPLES = 50

# 高质量分析 (更准确但较慢):
# NUM_BACKGROUND_SAMPLES = 100, NUM_TEST_SAMPLES = 500

# 内存受限 (使用CPU):
# DEVICE = 'cpu', NUM_BACKGROUND_SAMPLES = 30, NUM_TEST_SAMPLES = 100
"""

import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import shap
import h5py
import pandas as pd
from torch.utils.data import Dataset
from model import CNN, CNNTransformer

# Matplotlib 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 'SimHei' 是常用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class WindowDataset(Dataset):
    """与predict.py保持一致的数据集类"""
    def __init__(self, data_file, label_file, model_dir, use_custom_norm=True):
        # 加载数据
        with h5py.File(data_file, 'r') as f:
            self.windows = f['windows'][:].astype(np.float32)
            self.positions = f['positions'][:].astype(np.float32)
        
        # 加载标签文件并生成标签
        with h5py.File(label_file, 'r') as f:
            label_windows = f['windows'][:].astype(np.float32)
        
        # 生成标签：如果窗口中包含1则为正样本，否则为负样本
        self.labels = np.zeros(label_windows.shape[-1], dtype=np.int32)
        for i in range(label_windows.shape[-1]):
            if np.any(label_windows[..., i] == 1):
                self.labels[i] = 1
            else:
                self.labels[i] = -1
        
        # 转换数据维度顺序为 (N, C, H, W)
        self.windows = np.transpose(self.windows, (3, 2, 0, 1))  # (N, 11, 32, 32)
        
        # 加载标准化参数
        if use_custom_norm and os.path.exists(os.path.join(model_dir, 'normalization_params.pth')):
            print(f"使用自定义标准化参数文件: {os.path.join(model_dir, 'normalization_params.pth')}")
            norm_params = torch.load(os.path.join(model_dir, 'normalization_params.pth'))
            self.means = norm_params['mean']
            self.stds = norm_params['std']
        else:
            # 计算每个通道的均值和标准差
            self.means = []
            self.stds = []
            for i in range(self.windows.shape[1]):
                self.means.append(float(np.mean(self.windows[:, i])))
                self.stds.append(float(np.std(self.windows[:, i])))
        
        # 标准化数据
        for i in range(self.windows.shape[1]):
            self.windows[:, i] = (self.windows[:, i] - self.means[i]) / (self.stds[i] + 1e-8)
        
        # 转换为PyTorch张量
        self.windows = torch.FloatTensor(self.windows)
        self.positions = torch.FloatTensor(self.positions)
        self.labels = torch.LongTensor(self.labels)
        
        print(f"数据集大小: {len(self.windows)}")
        print(f"正样本数量: {torch.sum(self.labels == 1).item()}")
        print(f"负样本数量: {torch.sum(self.labels == -1).item()}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.positions[idx], self.labels[idx]


def _load_prediction_runtime_for_shap():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    prediction_dir = os.path.join(project_root, "模型预测")
    common_dir = os.path.join(project_root, "common")
    for path in (prediction_dir, common_dir, project_root):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    import importlib

    return importlib.import_module("predict")


def _load_dataset_for_shap(args, model_dir):
    use_prediction_dataset = bool(getattr(args, "use_prediction_dataset", False))
    if use_prediction_dataset:
        predict_runtime = _load_prediction_runtime_for_shap()
        selected_channels = None
        if hasattr(predict_runtime, "parse_selected_channels"):
            selected_channels = predict_runtime.parse_selected_channels(getattr(args, "selected_channels", ""))
        return predict_runtime.PredictionDataset(
            args.input,
            args.label_file,
            model_dir,
            use_custom_norm=True,
            norm_params_path=getattr(args, "norm_params_path", None),
            patch_size=getattr(args, "patch_size", None),
            patch_stride=getattr(args, "patch_stride", None),
            use_reflect_padding=bool(getattr(args, "reflect_padding", False)),
            selected_channels=selected_channels,
        )

    try:
        return WindowDataset(args.input, args.label_file, model_dir, use_custom_norm=True)
    except Exception:
        predict_runtime = _load_prediction_runtime_for_shap()
        return predict_runtime.PredictionDataset(
            args.input,
            args.label_file,
            model_dir,
            use_custom_norm=True,
            norm_params_path=getattr(args, "norm_params_path", None),
            patch_size=getattr(args, "patch_size", None),
            patch_stride=getattr(args, "patch_stride", None),
            use_reflect_padding=bool(getattr(args, "reflect_padding", False)),
        )


def _dataset_positions_array(dataset):
    positions = getattr(dataset, "positions", None)
    if positions is None:
        return np.empty((len(dataset), 2), dtype=np.float64)
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    return np.asarray(positions, dtype=np.float64)[:, :2]


def _coordinate_mask(prediction_positions, region_positions):
    prediction_positions = np.asarray(prediction_positions, dtype=np.float64)
    region_positions = np.asarray(region_positions, dtype=np.float64)
    if len(region_positions) == 0:
        return np.zeros(len(prediction_positions), dtype=bool)
    rounded_region = {
        (round(float(x), 6), round(float(y), 6))
        for x, y in region_positions[:, :2]
    }
    return np.asarray(
        [
            (round(float(x), 6), round(float(y), 6)) in rounded_region
            for x, y in prediction_positions[:, :2]
        ],
        dtype=bool,
    )


def _load_test_area_indices(dataset, area_file):
    positions = _dataset_positions_array(dataset)
    total_count = len(dataset)
    if not area_file:
        return np.arange(total_count, dtype=np.int64)
    if not os.path.exists(area_file):
        raise FileNotFoundError(f"测试区域文件不存在: {area_file}")

    suffix = os.path.splitext(area_file)[1].lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(area_file, "r") as handle:
            key = next((name for name in ("test_mask", "mask", "area_mask") if name in handle), None)
            if key is not None:
                mask = np.asarray(handle[key][:]).astype(bool).reshape(-1)
                if len(mask) != total_count:
                    raise ValueError(f"测试区域 mask 长度不匹配: mask={len(mask)}, samples={total_count}")
                return np.where(mask)[0].astype(np.int64)

            key = next((name for name in ("test_indices", "indices", "area_indices") if name in handle), None)
            if key is not None:
                indices = np.asarray(handle[key][:]).astype(np.int64).reshape(-1)
                return indices[(indices >= 0) & (indices < total_count)]

            key = next((name for name in ("positions", "coordinates", "coords") if name in handle), None)
            if key is not None:
                mask = _coordinate_mask(positions, np.asarray(handle[key][:], dtype=np.float64))
                return np.where(mask)[0].astype(np.int64)

        raise KeyError("测试区域 H5 需要包含 test_mask/test_indices/positions 等数据集。")

    if suffix in {".npy", ".npz"}:
        data = np.load(area_file, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            key = next((name for name in ("test_mask", "mask", "test_indices", "indices", "positions", "coordinates") if name in data), None)
            if key is None:
                raise KeyError("测试区域 NPZ 需要包含 test_mask/test_indices/positions 等数组。")
            array = np.asarray(data[key])
        else:
            array = np.asarray(data)
        if array.ndim == 1 and len(array) == total_count and np.isin(array, [0, 1, False, True]).all():
            return np.where(array.astype(bool))[0].astype(np.int64)
        if array.ndim == 1:
            indices = array.astype(np.int64)
            return indices[(indices >= 0) & (indices < total_count)]
        mask = _coordinate_mask(positions, array)
        return np.where(mask)[0].astype(np.int64)

    frame = pd.read_csv(area_file, sep=None, engine="python")
    if frame.shape[1] < 2:
        raise ValueError("测试区域坐标文件至少需要两列坐标。")
    coords = frame.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float64)
    mask = _coordinate_mask(positions, coords)
    return np.where(mask)[0].astype(np.int64)


def _sample_shap_indices(dataset, args):
    total_samples = len(dataset)
    use_test_area = bool(getattr(args, "use_test_area", False))
    if use_test_area:
        candidate_indices = _load_test_area_indices(dataset, getattr(args, "test_area_file", None))
        print(f"SHAP 测试集模式: 独立测试区样本数 {len(candidate_indices)}")
    else:
        candidate_indices = np.arange(total_samples, dtype=np.int64)

    if len(candidate_indices) < 2:
        raise ValueError("可用于 SHAP 的样本不足，至少需要 2 个样本。")

    rng = np.random.default_rng(42)
    candidate_indices = rng.permutation(candidate_indices)
    current_num_background_samples = min(int(args.num_background_samples), max(1, len(candidate_indices) // 2))
    remaining = len(candidate_indices) - current_num_background_samples
    current_num_test_samples = min(int(args.num_test_samples), remaining)
    if current_num_test_samples <= 0:
        current_num_background_samples = max(1, len(candidate_indices) - 1)
        current_num_test_samples = len(candidate_indices) - current_num_background_samples
    if current_num_test_samples <= 0:
        raise ValueError("没有可用于解释的测试样本，请减少背景样本数或提供更多测试区样本。")

    background_indices = candidate_indices[:current_num_background_samples]
    test_indices = candidate_indices[
        current_num_background_samples:current_num_background_samples + current_num_test_samples
    ]
    return background_indices, test_indices


def _clean_channel_name(raw_name, index):
    text = str(raw_name or "").strip()
    if not text:
        return f"Feature {index + 1}"
    text = os.path.basename(text)
    text = re.sub(r"\.(grd|tif|tiff|img|dat|csv|txt)$", "", text, flags=re.IGNORECASE)

    element_match = re.search(
        r"(?<![A-Za-z])(Ag|As|Au|Cu|Sb|Pb|Zn|Mo|W|Sn|Bi|Hg|Fe|Mn|Co|Ni|Cr|Cd|Ba|La|Ce|Y|U|Th)(?![a-z])",
        text,
    )
    if element_match:
        return element_match.group(1)

    vector_match = re.search(r"vector[_\s-]*dim[_\s-]*(\d+)", text, flags=re.IGNORECASE)
    if vector_match:
        return f"Feature {int(vector_match.group(1))}"

    if "ΔT" in text or "DeltaT" in text or "delta_t" in text.lower():
        return "Delta T"

    ascii_text = re.sub(r"[^0-9A-Za-z_+\-./ ]+", "", text).strip(" _-.")
    return ascii_text or f"Feature {index + 1}"


def _format_shap_display_name(name):
    text = str(name or "").strip()
    if text == "1_mod":
        return "FIS"
    if text in {"Delta T", "DeltaT", "delta_t"} or "ΔT" in text:
        return "ΔT"
    return text


def _get_shap_channel_names(dataset, channel_count):
    metadata = getattr(dataset, "metadata", {}) or {}
    raw_names = list(metadata.get("selected_channel_names") or metadata.get("available_channel_names") or [])
    if len(raw_names) != channel_count:
        raw_names = [f"Feature {index + 1}" for index in range(channel_count)]

    names = [_format_shap_display_name(_clean_channel_name(name, index)) for index, name in enumerate(raw_names)]
    seen = {}
    unique_names = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name} {seen[name]}")
    return unique_names


def _apply_custom_channel_names(channel_names, custom_names_text):
    channel_names = list(channel_names or [])
    text = str(custom_names_text or "").strip()
    if not text:
        return channel_names

    custom_names = [
        item.strip()
        for item in re.split(r"[;\n,，；]+", text)
        if item.strip()
    ]
    if not custom_names:
        return channel_names

    resolved = []
    for index, default_name in enumerate(channel_names):
        resolved.append(_format_shap_display_name(custom_names[index] if index < len(custom_names) else default_name))

    seen = {}
    unique_names = []
    for name in resolved:
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name} {seen[name]}")
    return unique_names


def _get_model_expectations(model: nn.Module):
    channels = getattr(model, "input_channels", None)
    if channels is None:
        channels = getattr(model, "channels", None)

    height = getattr(model, "img_size", None)
    width = getattr(model, "img_size", None)
    height = getattr(model, "img_height", height)
    width = getattr(model, "img_width", width)

    return channels, height, width


def _unwrap_model_state(loaded_obj):
    """Extract actual state_dict and any checkpoint metadata."""
    metadata = {}
    state_dict = loaded_obj
    if isinstance(loaded_obj, dict) and "model_state" in loaded_obj:
        metadata = {k: v for k, v in loaded_obj.items() if k != "model_state"}
        state_dict = loaded_obj["model_state"]
        print("检测到包含模型元数据的 checkpoint，已提取 model_state。")
    return state_dict, metadata


def _reshape_flattened_input(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    channels, height, width = _get_model_expectations(model)
    if x.dim() == 2 and all(v is not None for v in (channels, height, width)):
        expected_features = channels * height * width
        if x.size(1) == expected_features:
            return x.view(x.size(0), channels, height, width)
    return x


class ShapModelWrapper(nn.Module):
    """确保模型输出至少二维，便于 SHAP 解释器处理二分类输出。"""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = _reshape_flattened_input(x, self.base_model)
        outputs = self.base_model(x)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(-1)
        return outputs


def align_with_model(tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Reshape or resize batched tensors to the model's expected spatial dimensions."""
    if tensor.dim() != 4:
        return tensor

    expected_channels, expected_height, expected_width = _get_model_expectations(model)
    if expected_channels is None:
        expected_channels = tensor.shape[1]
    if expected_height is None:
        expected_height = tensor.shape[2]
    if expected_width is None:
        expected_width = tensor.shape[3]

    if tensor.shape[1] != expected_channels:
        raise ValueError(
            f"输入数据通道数 {tensor.shape[1]} 与模型期望的 {expected_channels} 不一致，无法自动适配"
        )

    if tensor.shape[2] != expected_height or tensor.shape[3] != expected_width:
        original_h, original_w = tensor.shape[2], tensor.shape[3]
        tensor = F.interpolate(
            tensor,
            size=(expected_height, expected_width),
            mode="bilinear",
            align_corners=False,
        )
        print(
            f"已自动将输入空间尺寸从 {original_h}x{original_w} 调整为模型期望的 {expected_height}x{expected_width}"
        )

    return tensor


def _create_cnn_candidate(prior, in_channels, height, width):
    input_dim = int(in_channels * height * width)
    return CNN(prior, input_dim, input_shape=(in_channels, height, width))


def _infer_cnn_from_checkpoint(prior, saved_state, data_shape=None):
    conv1 = saved_state.get("conv1.weight")
    fc1 = saved_state.get("fc1.weight")
    if conv1 is None or fc1 is None:
        raise ValueError("checkpoint 缺少 CNN 所需的 conv1 或 fc1 权重")

    in_channels = conv1.shape[1]
    target_fc_features = fc1.shape[1]
    candidate_shapes = []

    if data_shape is not None and data_shape[0] == in_channels:
        candidate_shapes.append((data_shape[1], data_shape[2]))

    if target_fc_features % 128 == 0:
        final_area = target_fc_features // 128
        sqrt_val = int(final_area ** 0.5)
        if sqrt_val * sqrt_val == final_area:
            final_sizes = [(sqrt_val, sqrt_val)]
        else:
            final_sizes = []
            for h in range(1, int(final_area ** 0.5) + 1):
                if final_area % h == 0:
                    w = final_area // h
                    final_sizes.append((h, w))
        for final_h, final_w in final_sizes:
            for pools in range(3, -1, -1):
                height = final_h * (2 ** pools)
                width = final_w * (2 ** pools)
                candidate_shapes.append((height, width))

    tried = set()
    for height, width in candidate_shapes:
        key = (height, width)
        if min(height, width) <= 0 or key in tried:
            continue
        tried.add(key)
        try:
            model = _create_cnn_candidate(prior, in_channels, height, width)
        except Exception as exc:
            print(f"跳过候选输入尺寸 {height}x{width}: {exc}")
            continue
        fc_features = model.fc1.weight.shape[1]
        if fc_features != target_fc_features:
            continue
        print(f"已根据 checkpoint 推断 CNN 输入尺寸: {height}x{width}, 通道数 {in_channels}")
        model.channels = in_channels
        return model

    raise ValueError("无法根据 checkpoint 推断 CNN 输入尺寸，请确认模型文件")


def _create_cnnt_from_checkpoint(prior, saved_state, input_dim, checkpoint_meta=None, data_shape=None):
    checkpoint_meta = checkpoint_meta or {}

    try:
        model = CNNTransformer.create_compatible_model(prior, saved_state)
        print("已根据检查点推断 CNNTransformer 结构")
        return model
    except Exception as exc:
        print(f"根据检查点推断 CNNTransformer 结构失败: {exc}")
        print("回退到根据数据维度创建 CNNTransformer")

    def _meta_int(key):
        value = checkpoint_meta.get(key)
        if isinstance(value, torch.Tensor):
            value = int(value.item())
        return value

    fallback_channels = _meta_int("input_channels")
    fallback_height = _meta_int("input_height")
    fallback_width = _meta_int("input_width")

    if data_shape is not None:
        if fallback_channels is None:
            fallback_channels = data_shape[0]
        if fallback_height is None:
            fallback_height = data_shape[1]
        if fallback_width is None:
            fallback_width = data_shape[2]

    if fallback_channels and fallback_height and fallback_width:
        inferred_input_dim = fallback_channels * fallback_height * fallback_width
    else:
        inferred_input_dim = input_dim

    model = CNNTransformer(
        prior,
        inferred_input_dim,
        input_channels=fallback_channels or (data_shape[0] if data_shape else 11),
    )

    # 保留高度和宽度信息，便于后续自动适配
    if fallback_height:
        model.img_size = fallback_height
        model.img_height = fallback_height
    if fallback_width:
        model.img_width = fallback_width

    return model


def load_model(model_type, model_path, prior, input_dim, device, data_shape=None):
    """根据模型类型加载训练好的深度模型。"""
    print(f"正在加载模型 ({model_type}): {model_path}")

    loaded_obj = torch.load(model_path, map_location=device)
    saved_state, checkpoint_meta = _unwrap_model_state(loaded_obj)

    if model_type == "cnn":
        model = _infer_cnn_from_checkpoint(prior, saved_state, data_shape)
    elif model_type == "cnnt":
        model = _create_cnnt_from_checkpoint(prior, saved_state, input_dim, checkpoint_meta, data_shape)
        saved_state = ensure_transformer_biases(saved_state, model)
    else:
        raise ValueError(f"暂不支持的模型类型: {model_type}")

    model = model.to(device)

    try:
        if hasattr(model, 'load_my_state_dict'):
            model.load_my_state_dict(saved_state)
            print("模型状态已使用 load_my_state_dict 加载")
        else:
            model.load_state_dict(saved_state)
            print("模型状态已使用 load_state_dict 加载")
    except RuntimeError as e:
        print(f"警告: 加载模型状态失败: {e}")
        print("尝试使用 strict=False 加载...")
        try:
            model.load_state_dict(saved_state, strict=False)
            print("模型状态已使用 strict=False 加载")
        except Exception as e2:
            print(f"使用 strict=False 加载模型状态时出错: {e2}")
            return None

    model.eval()
    return model


def ensure_transformer_biases(saved_state, model):
    """为缺失的 Transformer 偏置项补零，兼容旧 checkpoint。"""
    required_keys = [
        "transformer.attn.in_proj_bias",
        "transformer.attn.out_proj.bias",
        "transformer.mlp.0.bias",
        "transformer.mlp.2.bias",
    ]
    model_state = model.state_dict()
    for key in required_keys:
        if key not in saved_state and key in model_state:
            saved_state[key] = torch.zeros_like(model_state[key])
            print(f"已为缺失参数 {key} 补充默认值")
    return saved_state

def explain_cnnt_with_shap(args):
    """为训练好的 CNN/CNNTransformer 模型生成 SHAP 解释。"""
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")

    # 获取模型目录
    model_dir = os.path.dirname(args.model_path)
    print(f"模型目录: {model_dir}")

    # 1. 加载数据集
    print("正在加载数据集...")
    try:
        dataset = _load_dataset_for_shap(args, model_dir)
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return

    # 从数据中获取实际输入维度
    sample_data = dataset[0][0]
    input_channels = sample_data.shape[0]
    input_height = sample_data.shape[1]
    input_width = sample_data.shape[2]
    actual_input_dim = input_channels * input_height * input_width
    print(f"检测到实际输入维度: {actual_input_dim} ({input_channels}×{input_height}×{input_width})")

    model_type = getattr(args, 'model_type', 'cnnt').lower()
    print(f"当前 SHAP 分析模型类型: {model_type}")

    # 2. 加载模型
    if model_type in {"cntt", "pucnn", "pucnnt", "pucnntransformer"}:
        predict_runtime = _load_prediction_runtime_for_shap()
        model, _ = predict_runtime.load_model(
            model_type,
            args.model_path,
            args.prior,
            actual_input_dim,
            device,
            data_shape=sample_data.shape,
        )
    else:
        model = load_model(
            model_type,
            args.model_path,
            args.prior,
            actual_input_dim,
            device,
            data_shape=sample_data.shape,
        )
    if model is None:
        print("模型加载失败，退出")
        return
    shap_model = ShapModelWrapper(model).to(device)

    # 3. 准备 SHAP 所需数据
    total_samples = len(dataset)
    current_num_background_samples = min(args.num_background_samples, total_samples // 2)
    if current_num_background_samples == 0:
        print("错误：没有足够的样本作为背景数据")
        return
    current_num_test_samples = min(args.num_test_samples, total_samples - current_num_background_samples)
    if current_num_test_samples == 0:
        print("错误：没有测试样本可供解释")
        return
    print(f"将使用 {current_num_background_samples} 个背景样本和 {current_num_test_samples} 个测试样本")
    indices = np.random.permutation(total_samples)
    background_indices = indices[:current_num_background_samples]
    test_indices = indices[current_num_background_samples:current_num_background_samples + current_num_test_samples]
    try:
        background_indices, test_indices = _sample_shap_indices(dataset, args)
        print(f"将使用 {len(background_indices)} 个背景样本和 {len(test_indices)} 个测试样本")
    except Exception as e:
        print(f"准备 SHAP 样本时出错: {e}")
        return
    background_data_img = torch.stack([dataset[i][0] for i in background_indices]).to(device)
    test_data_img = torch.stack([dataset[i][0] for i in test_indices]).to(device)
    background_data_img = align_with_model(background_data_img, model)
    test_data_img = align_with_model(test_data_img, model)
    background_data = background_data_img.view(background_data_img.size(0), -1)
    test_data = test_data_img.view(test_data_img.size(0), -1)
    test_labels = torch.stack([dataset[i][2] for i in test_indices])
    print(f"背景数据形状: {background_data_img.shape}")
    print(f"测试数据形状: {test_data_img.shape}")

    # 4. 创建 SHAP Explainer 并计算 SHAP 值
    print("正在初始化 SHAP Explainer...")
    explainer = None
    explainer_type = None
    try:
        print("尝试 GradientExplainer...")
        explainer = shap.GradientExplainer(shap_model, background_data)
        explainer_type = "GradientExplainer"
        print("✅ 使用 GradientExplainer")
    except Exception as e:
        print(f"❌ GradientExplainer 初始化失败: {e}")
    if explainer is None:
        try:
            print("尝试 DeepExplainer...")
            explainer = shap.DeepExplainer(shap_model, background_data)
            explainer_type = "DeepExplainer"
            print("✅ 使用 DeepExplainer")
        except Exception as e:
            print(f"❌ DeepExplainer 初始化失败: {e}")
    if explainer is None:
        try:
            print("尝试 KernelExplainer (可能较慢)...")
            def model_predict(x):
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(device)
                with torch.no_grad():
                    outputs = shap_model(x)
                    return outputs.cpu().numpy().flatten()
            background_data_np = background_data.cpu().numpy()
            explainer = shap.KernelExplainer(model_predict, background_data_np)
            explainer_type = "KernelExplainer"
            print("✅ 使用 KernelExplainer")
        except Exception as e:
            print(f"❌ KernelExplainer 初始化失败: {e}")
            print("所有SHAP解释器都失败，无法继续")
            return

    # 统一的 SHAP 值计算流程
    print(f"正在为 {test_data.shape[0]} 个测试样本计算 SHAP 值...")
    try:
        if explainer_type == "KernelExplainer":
            test_data_np = test_data.cpu().numpy()
            shap_values_output = explainer.shap_values(test_data_np, nsamples=args.kernel_nsamples)
        else:
            shap_values_output = explainer.shap_values(test_data)
        if isinstance(shap_values_output, list):
            shap_values_np = shap_values_output[0]
        else:
            shap_values_np = shap_values_output
        if isinstance(shap_values_np, torch.Tensor):
            shap_values_np = shap_values_np.cpu().detach().numpy()
        if shap_values_np.shape[-1] == 1:
            shap_values_np = np.squeeze(shap_values_np, axis=-1)

        n_samples, n_channels, height, width = test_data_img.shape
        expected_flat_dim = n_channels * height * width
        if shap_values_np.ndim == 2 and shap_values_np.shape[1] == expected_flat_dim:
            shap_values_np = shap_values_np.reshape(n_samples, n_channels, height, width)
            print(f"已将 SHAP 值重塑为图像形状: {shap_values_np.shape}")
        elif shap_values_np.ndim != 4:
            print(f"警告: 当前 SHAP 值形状 {shap_values_np.shape} 无法映射到 (N,C,H,W)，后续图像可视化可能被跳过")
        print(f"✅ SHAP 值计算完成，形状: {shap_values_np.shape}")
        print(f"使用的解释器: {explainer_type}")
    except Exception as e:
        print(f"❌ 计算 SHAP 值时出错: {e}")
        print(f"使用的解释器: {explainer_type}")
        return

    # 只保留通道重要性条形图和SHAP summary plot
    shap_output_dir = getattr(args, 'output_dir', None)
    if shap_output_dir:
        shap_output_dir = os.path.abspath(shap_output_dir)
    else:
        shap_output_dir = os.path.join(model_dir, 'shap_explanations')
    os.makedirs(shap_output_dir, exist_ok=True)
    print(f"SHAP 结果将保存到: {shap_output_dir}")
    test_data_np = test_data_img.cpu().numpy()
    channel_names = _get_shap_channel_names(dataset, test_data_np.shape[1])
    channel_names = _apply_custom_channel_names(channel_names, getattr(args, "custom_channel_names", ""))
    importance_xlabel = str(getattr(args, "importance_xlabel", "") or "Mean |SHAP|")
    summary_xlabel = str(getattr(args, "summary_xlabel", "") or "SHAP value (impact on model output)")
    channel_shap_values_2d = None
    channel_feature_values_2d = None
    channel_importance = None
    sorted_idx = None
    channel_names_sorted = None
    if shap_values_np.ndim == 4:
        # Treat each image channel as one grouped feature. Both plots use this
        # same signed channel-level SHAP matrix before taking absolute values.
        channel_shap_values_2d = np.mean(shap_values_np, axis=(2, 3))
        channel_feature_values_2d = np.mean(test_data_np, axis=(2, 3))
        channel_importance = np.mean(np.abs(channel_shap_values_2d), axis=0)
        sorted_idx = np.argsort(channel_importance)[::-1]
        channel_names_sorted = [channel_names[i] for i in sorted_idx]

        try:
            summary_values_path = os.path.join(shap_output_dir, 'shap_summary_values.csv')
            test_labels_np = test_labels.detach().cpu().numpy().reshape(-1)
            rows = []
            for sample_order, dataset_index in enumerate(np.asarray(test_indices, dtype=np.int64).reshape(-1)):
                label_value = int(test_labels_np[sample_order]) if sample_order < len(test_labels_np) else ""
                for channel_index, channel_name in enumerate(channel_names):
                    rows.append(
                        {
                            "sample_order": int(sample_order),
                            "dataset_index": int(dataset_index),
                            "label": label_value,
                            "channel": channel_name,
                            "channel_index": int(channel_index),
                            "shap_value": float(channel_shap_values_2d[sample_order, channel_index]),
                            "feature_value": float(channel_feature_values_2d[sample_order, channel_index]),
                        }
                    )
            pd.DataFrame(rows).to_csv(summary_values_path, index=False, encoding="utf-8-sig")
            print(f"SHAP summary明细已保存至: {summary_values_path}")
        except Exception as e:
            print(f"保存SHAP summary明细时出错: {e}")
    # 1. 通道重要性条形图（横向，按重要性降序排列）
    try:
        if shap_values_np.ndim != 4:
            raise ValueError("当前 SHAP 值不是四维张量，无法计算通道重要性")
        channel_importance_sorted = channel_importance[sorted_idx]
        importance_csv_path = os.path.join(shap_output_dir, 'shap_channel_importance.csv')
        pd.DataFrame(
            {
                "channel": channel_names,
                "importance": channel_importance,
            }
        ).to_csv(importance_csv_path, index=False, encoding="utf-8-sig")
        print(f"SHAP通道重要性数值已保存至: {importance_csv_path}")
        plt.figure(figsize=args.channel_importance_figsize)
        plt.barh(channel_names_sorted, channel_importance_sorted, alpha=0.7, color='skyblue')
        plt.gca().invert_yaxis()
        plt.xlabel(importance_xlabel)
        channel_plot_path = os.path.join(shap_output_dir, 'shap_channel_importance.png')
        plt.tight_layout()
        plt.savefig(channel_plot_path, dpi=args.figure_dpi, bbox_inches='tight')
        plt.close()
        print(f"通道重要性图已保存至 {channel_plot_path}")
    except Exception as e:
        print(f"生成通道重要性图时出错: {e}")

    # 2. SHAP summary plot
    try:
        if shap_values_np.ndim != 4:
            raise ValueError("当前 SHAP 值不是四维张量，无法生成 summary plot")
        # 统一列顺序，确保与条形图排序完全一致
        feature_values_2d = channel_feature_values_2d[:, sorted_idx]
        shap_values_2d = channel_shap_values_2d[:, sorted_idx]

        shap.summary_plot(
            shap_values_2d,
            feature_values_2d,
            feature_names=channel_names_sorted,
            show=False,
            plot_type="dot",
            color_bar=True,
            max_display=len(channel_names_sorted),
            sort=False,
        )
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("black")
        ax.set_xlabel("")
        plt.gcf().set_size_inches(*args.summary_plot_figsize)
        summary_plot_path = os.path.join(shap_output_dir, 'shap_summary_plot.png')
        plt.savefig(summary_plot_path, dpi=args.figure_dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"SHAP Summary Plot已保存至: {summary_plot_path}")
    except Exception as e:
        print(f"生成SHAP Summary Plot时出错: {e}")

    print(f"\n✅ SHAP 分析完成！")
    print(f"📁 结果已保存到: {shap_output_dir}")
    print(f"🔧 使用的解释器: {explainer_type}")

def main():
    # =================== 手动配置区域 - 在这里调整你的参数 ===================
    # 
    # 📋 使用说明：
    # 1. 直接修改下面的参数值，无需使用命令行
    # 2. 修改后保存文件，然后运行: python evaluate_shap_cnnt.py
    # 3. 参数说明中的建议值范围仅供参考，可以根据实际需要调整
    # 4. 如果GPU内存不足，可以减少样本数量或改用CPU
    # 5. 如果想跳过某些分析，可以将对应的ENABLE参数设置为False
    #
    # =====================================================================
    
    # 🔧 基本文件路径配置
    MODEL_PATH = 'result/cnnt-9-0.2-10%/model.pth'       # 修改为你的模型路径
    MODEL_TYPE = 'cnnt'                                   # 可选: 'cnn' 或 'cnnt'
    INPUT_DATA = 'data/windows_combined_data.h5'          # 修改为你的输入数据路径
    LABEL_FILE = 'data/deposit_labels.h5'                # 修改为你的标签文件路径
    
    # 🔧 SHAP 分析核心参数
    NUM_BACKGROUND_SAMPLES = 50    # 背景样本数量 (建议: 20-200, 越大越准确但越慢)
    NUM_TEST_SAMPLES = 200         # 测试样本数量 (建议: 50-1000, 越大分析越全面)
    PRIOR = 0.2                    # 先验概率值 (根据你的数据集调整)
    
    # 🔧 设备配置
    DEVICE = 'cuda'               # 使用的设备: 'cuda' 或 'cpu'
    
    # 🔧 可视化参数 (可选调整)
    MAX_IMAGES_TO_PLOT = 3        # 生成图像解释的最大样本数 (建议: 1-10)
    MAX_CHANNELS_TO_SHOW = 3      # 图像中显示的最大通道数 (建议: 1-5)
    TOP_CHANNELS_FOR_ANALYSIS = 6 # 详细分析的顶级通道数 (建议: 3-10)
    
    # 🔧 图表尺寸和质量配置
    FIGURE_DPI = 300              # 图片分辨率 (建议: 150-300)
    GLOBAL_IMPORTANCE_FIGSIZE = (10, 8)     # 全局重要性图尺寸
    CHANNEL_IMPORTANCE_FIGSIZE = (12, 6)    # 通道重要性图尺寸
    COMPARISON_FIGSIZE = (12, 6)            # 正负样本对比图尺寸
    SUMMARY_PLOT_FIGSIZE = (10, 8)          # 总结图尺寸
    SCATTER_3D_FIGSIZE = (12, 9)            # 3D散点图尺寸
    
    # 🔧 分析行为控制
    ENABLE_IMAGE_PLOTS = True     # 是否生成图像解释图 (True/False)
    ENABLE_SCATTER_PLOTS = True   # 是否生成散点图分析 (True/False)
    ENABLE_SUMMARY_PLOTS = True   # 是否生成总结图 (True/False)
    ENABLE_3D_PLOTS = True        # 是否生成3D图 (True/False)
    
    # 🔧 KernelExplainer 参数 (仅在其他解释器失败时使用)
    KERNEL_NSAMPLES = 100         # 采样数量 (建议: 50-500, 越大越准确但越慢)
    
    # =====================================================================
    
    # 创建参数对象来替代命令行参数
    class Args:
        def __init__(self):
            self.model_path = MODEL_PATH
            self.model_type = MODEL_TYPE
            self.input = INPUT_DATA
            self.label_file = LABEL_FILE
            self.prior = PRIOR
            self.num_background_samples = NUM_BACKGROUND_SAMPLES
            self.num_test_samples = NUM_TEST_SAMPLES
            self.device = DEVICE
            # 添加可视化参数
            self.max_images_to_plot = MAX_IMAGES_TO_PLOT
            self.max_channels_to_show = MAX_CHANNELS_TO_SHOW
            self.top_channels_for_analysis = TOP_CHANNELS_FOR_ANALYSIS
            self.kernel_nsamples = KERNEL_NSAMPLES
            # 添加图表配置参数
            self.figure_dpi = FIGURE_DPI
            self.global_importance_figsize = GLOBAL_IMPORTANCE_FIGSIZE
            self.channel_importance_figsize = CHANNEL_IMPORTANCE_FIGSIZE
            self.comparison_figsize = COMPARISON_FIGSIZE
            self.summary_plot_figsize = SUMMARY_PLOT_FIGSIZE
            self.scatter_3d_figsize = SCATTER_3D_FIGSIZE
            # 添加行为控制参数
            self.enable_image_plots = ENABLE_IMAGE_PLOTS
            self.enable_scatter_plots = ENABLE_SCATTER_PLOTS
            self.enable_summary_plots = ENABLE_SUMMARY_PLOTS
            self.enable_3d_plots = ENABLE_3D_PLOTS
    
    args = Args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在 - {args.model_path}")
        return
    
    # 检查数据文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 数据文件不存在 - {args.input}")
        return
        
    if not os.path.exists(args.label_file):
        print(f"错误: 标签文件不存在 - {args.label_file}")
        return
    
    print("开始 SHAP 分析...")
    print(f"📁 模型路径: {args.model_path}")
    print(f"📁 数据文件: {args.input}")
    print(f"📁 标签文件: {args.label_file}")
    print(f"🔢 背景样本数: {args.num_background_samples}")
    print(f"🔢 测试样本数: {args.num_test_samples}")
    print(f"🖥️  使用设备: {args.device}")
    print("-" * 50)
    explain_cnnt_with_shap(args)

if __name__ == "__main__":
    main()
