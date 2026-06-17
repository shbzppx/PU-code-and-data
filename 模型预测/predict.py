import argparse
import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(CURRENT_DIR)
COMMON_DIR = os.path.join(CODE_ROOT, "common")
for path in (CODE_ROOT, COMMON_DIR):
    if path not in sys.path:
        sys.path.append(path)

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from feature_channel_utils import (
    describe_selected_channels,
    infer_h5_channel_names,
    parse_selected_channels,
    subset_samples_by_channels,
)

from model import (LinearClassifier, ThreeLayerPerceptron, MultiLayerPerceptron,
                   CNN, CNNTransformer, CNNTokenTransformer, RandomForestBinaryClassifier,
                   OneClassSVMClassifier, PURandomForestClassifier, TwoStepPULearning)
import h5py
from datetime import datetime
import pickle # <-- 确保在文件顶部导入 pickle
try:
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    import matplotlib
    # 设置中文字体支持
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError as e:
    print(f"错误: 缺少必要的库 - {e}")
    print("请安装必要的库: pip install pandas openpyxl matplotlib")
    exit(1)


def _torch_load_compat(path, map_location=None):
    """Load trusted local PyTorch artifacts across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _normalize_coord_offset(coord_offset):
    if coord_offset is None:
        return 0.0, 0.0
    if isinstance(coord_offset, (int, float, np.integer, np.floating)):
        value = float(coord_offset)
        return value, value
    if len(coord_offset) == 1:
        value = float(coord_offset[0])
        return value, value
    return float(coord_offset[0]), float(coord_offset[1])


try:
    from feature.patch_creator import PatchCreator
except Exception:  # pragma: no cover
    PatchCreator = None


def _read_mineral_points(label_path):
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"矿点文件不存在: {label_path}")

    ext = os.path.splitext(label_path)[1].lower()
    if ext not in {".txt", ".csv", ".tsv", ".dat"}:
        raise ValueError("矿点标签文件必须是 txt/dat/csv/tsv。")

    frame = None
    last_error = None
    for kwargs in (
        {"sep": None, "engine": "python"},
        {"sep": "\t"},
        {"sep": ","},
    ):
        try:
            frame = pd.read_csv(label_path, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    if frame is None:
        raise last_error

    if frame.empty:
        raise ValueError("矿点文件为空。")

    column_map = {str(col).strip().lower(): col for col in frame.columns}
    x_column = next((column_map[key] for key in ("x", "coord_x", "point_x", "east", "easting", "longitude") if key in column_map), None)
    y_column = next((column_map[key] for key in ("y", "coord_y", "point_y", "north", "northing", "latitude") if key in column_map), None)
    if x_column is None or y_column is None:
        if frame.shape[1] >= 2:
            x_column, y_column = frame.columns[:2].tolist()
        else:
            raise KeyError("矿点文件至少需要两列坐标。")

    result = frame[[x_column, y_column]].copy()
    result.columns = ["x", "y"]
    result["x"] = pd.to_numeric(result["x"], errors="coerce")
    result["y"] = pd.to_numeric(result["y"], errors="coerce")
    result = result.dropna(subset=["x", "y"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("矿点文件中没有可用的 X/Y 坐标。")
    return result


def _extract_metadata(handle):
    metadata = {}
    if "metadata" in handle:
        metadata.update(dict(handle["metadata"].attrs))
    for key in ("x_min", "x_max", "y_min", "y_max", "nx", "ny", "window_width", "window_height"):
        if key in handle.attrs:
            metadata[key] = handle.attrs[key]
    return metadata


def _patch_indices_to_geo(coords, metadata):
    if coords is None:
        return None

    arr = np.asarray(coords)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        return arr.astype(np.float64, copy=False)

    meta = metadata or {}
    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", meta.get("image_width", meta.get("width", 0))))
        ny = int(meta.get("ny", meta.get("image_height", meta.get("height", 0))))
    except (KeyError, TypeError, ValueError):
        return arr.astype(np.float64, copy=False)

    if nx <= 1 or ny <= 1:
        return arr.astype(np.float64, copy=False)

    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span == 0 or y_span == 0:
        return arr.astype(np.float64, copy=False)

    direct_x = arr[:, 0]
    direct_y = arr[:, 1]
    x_bound = max(nx - 1, 1) * 1.5
    y_bound = max(ny - 1, 1) * 1.5
    direct_ok = np.nanmax(np.abs(direct_x)) <= x_bound and np.nanmax(np.abs(direct_y)) <= y_bound

    swapped_x = arr[:, 1]
    swapped_y = arr[:, 0]
    swapped_ok = np.nanmax(np.abs(swapped_x)) <= x_bound and np.nanmax(np.abs(swapped_y)) <= y_bound

    if not direct_ok and swapped_ok:
        x_values = swapped_x
        y_values = swapped_y
    else:
        x_values = direct_x
        y_values = direct_y

    if np.nanmax(np.abs(x_values)) <= x_bound and np.nanmax(np.abs(y_values)) <= y_bound:
        x_step = x_span / max(nx - 1, 1)
        y_step = y_span / max(ny - 1, 1)
        coordinates_are_centers = bool(meta.get("coordinates_are_centers", False))
        window_width = int(meta.get("window_width", 1) or 1)
        window_height = int(meta.get("window_height", 1) or 1)
        x_offset = 0.0 if coordinates_are_centers else ((window_width - 1) / 2.0 if window_width > 1 else 0.0)
        y_offset = 0.0 if coordinates_are_centers else ((window_height - 1) / 2.0 if window_height > 1 else 0.0)

        geo_x = x_min + (x_values + x_offset) * x_step
        geo_y = y_max - (y_values + y_offset) * y_step
        return np.column_stack([geo_x, geo_y]).astype(np.float64, copy=False)

    return arr.astype(np.float64, copy=False)


def _window_contains_minerals(sample_coords, minerals, metadata, patch_size):
    labels = np.zeros(len(sample_coords), dtype=np.int32)
    if minerals is None or len(minerals) == 0:
        return labels

    coords = np.asarray(sample_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return labels

    meta = metadata or {}
    mineral_coords = minerals[["x", "y"]].to_numpy(dtype=np.float64)

    try:
        x_min = float(meta["x_min"])
        x_max = float(meta["x_max"])
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        nx = int(meta.get("nx", meta.get("image_width", meta.get("width", 0))))
        ny = int(meta.get("ny", meta.get("image_height", meta.get("height", 0))))
    except (KeyError, TypeError, ValueError):
        x_min = x_max = y_min = y_max = 0.0
        nx = ny = 0

    if nx > 1 and ny > 1 and (x_max - x_min) != 0 and (y_max - y_min) != 0:
        x_step = (x_max - x_min) / max(nx - 1, 1)
        y_step = (y_max - y_min) / max(ny - 1, 1)
        window_width = int(patch_size or meta.get("window_width", 1) or 1)
        window_height = int(patch_size or meta.get("window_height", 1) or 1)
        rows = coords[:, 0]
        cols = coords[:, 1]
        coordinates_are_centers = bool(meta.get("coordinates_are_centers", False))
        if coordinates_are_centers:
            center_x = x_min + cols * x_step
            center_y = y_max - rows * y_step
            half_width = max(float(window_width) * x_step / 2.0, 0.0)
            half_height = max(float(window_height) * y_step / 2.0, 0.0)
            x_left = center_x - half_width
            x_right = center_x + half_width
            y_top = center_y + half_height
            y_bottom = center_y - half_height
        else:
            x_left = x_min + cols * x_step
            x_right = x_left + window_width * x_step
            y_top = y_max - rows * y_step
            y_bottom = y_top - window_height * y_step

        for mineral_x, mineral_y in mineral_coords:
            inside = (mineral_x >= x_left) & (mineral_x < x_right) & (mineral_y <= y_top) & (mineral_y > y_bottom)
            labels[inside] = 1
        return labels

    geo_coords = _patch_indices_to_geo(coords, meta)
    if geo_coords is None or len(geo_coords) == 0:
        return labels

    x_scale = abs(float(meta.get("x_scale", 1.0) or 1.0))
    y_scale = abs(float(meta.get("y_scale", 1.0) or 1.0))
    window_width = int(meta.get("window_width", patch_size) or patch_size or 1)
    window_height = int(meta.get("window_height", patch_size) or patch_size or 1)
    half_width = max(float(window_width) * x_scale / 2.0, 0.0)
    half_height = max(float(window_height) * y_scale / 2.0, 0.0)
    for mineral_x, mineral_y in mineral_coords:
        inside = (
            (np.abs(geo_coords[:, 0] - mineral_x) <= half_width)
            & (np.abs(geo_coords[:, 1] - mineral_y) <= half_height)
        )
        labels[inside] = 1
    return labels


def _load_feature_tensor(data_path, patch_size, patch_stride, enable_reflect_padding=False, selected_channels=None):
    with h5py.File(data_path, "r") as handle:
        metadata = _extract_metadata(handle)
        metadata["available_channel_names"] = infer_h5_channel_names(data_path)
        if "windows" in handle:
            windows = np.asarray(handle["windows"][:], dtype=np.float32)
            coordinates = np.asarray(handle["positions"][:], dtype=np.float64) if "positions" in handle else None
            if coordinates is None and "coordinates" in handle:
                coordinates = np.asarray(handle["coordinates"][:], dtype=np.float64)
            if windows.ndim < 4:
                raise ValueError(f"windows 数据至少应为 4 维，当前为 {windows.shape}")
            samples = np.transpose(windows, (3, 2, 0, 1))
            samples, metadata = subset_samples_by_channels(samples, metadata, selected_channels)
            return samples, coordinates, metadata, "windows"

    if PatchCreator is None:
        raise ImportError("PatchCreator is unavailable; cannot generate patches from raw H5.")

    creator = PatchCreator(data_path)
    try:
        samples, coordinates = creator.generate_patches(
            int(patch_size),
            int(patch_stride),
            enable_padding=bool(enable_reflect_padding),
            padding_mode="reflect",
        )
    finally:
        creator.close()
    metadata = dict(metadata)
    metadata["window_width"] = int(patch_size)
    metadata["window_height"] = int(patch_size)
    metadata["patch_stride"] = int(patch_stride)
    metadata["prediction_reflect_padding"] = bool(enable_reflect_padding)
    metadata["coordinates_are_centers"] = bool(enable_reflect_padding)
    samples, metadata = subset_samples_by_channels(np.asarray(samples, dtype=np.float32), metadata, selected_channels)
    return samples, np.asarray(coordinates, dtype=np.float64), metadata, "patches"

def load_model(model_name, model_path, prior, input_dim, device, data_shape=None):
    """加载训练好的模型并根据 checkpoint 元数据校验输入形状"""
    models = {
        "linear": LinearClassifier,
        "3lp": ThreeLayerPerceptron,
        "mlp": MultiLayerPerceptron,
        "cnn": CNN,
        "cnnt": CNNTransformer,
        "cntt": CNNTokenTransformer,
        "pucnn": CNN,
        "pucnnt": CNNTransformer,
        "pucnntransformer": CNNTokenTransformer,
        # RF / OCSVM / 2step / PU-Random Forest 将单独处理
    }
    
    # --- 处理使用 pickle 保存的模型 --- 
    use_pickle = model_name in {"rf", "ocsvm", "2step", "purf"}
    attempted_pickle = False
    if use_pickle:
        try:
            attempted_pickle = True
            if not model_path.lower().endswith('.pkl'):
                raise pickle.UnpicklingError("模型文件扩展名不是 .pkl，跳过 pickle 加载")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"成功从 pickle 文件加载 {model_name.upper()} 模型: {model_path}")
            
            # 可选：进行类型检查
            if model_name == "rf" and not isinstance(model, RandomForestBinaryClassifier):
                 print(f"警告: 加载的模型类型 ({type(model).__name__}) 与预期的 RF 不符。")
            elif model_name == "ocsvm" and not isinstance(model, OneClassSVMClassifier):
                 print(f"警告: 加载的模型类型 ({type(model).__name__}) 与预期的 OCSVM 不符。")
            elif model_name == "2step" and not isinstance(model, TwoStepPULearning):
                 print(f"警告: 加载的模型类型 ({type(model).__name__}) 与预期的 TwoStepPULearning 不符。")
            elif model_name == "purf" and not isinstance(model, PURandomForestClassifier):
                 print(f"警告: 加载的模型类型 ({type(model).__name__}) 与预期的 PURandomForestClassifier 不符。")

            # 根据需要设置设备或 is_fitted 状态
            if hasattr(model, 'device'):
                 model.device = device 
            if hasattr(model, 'is_fitted'):
                 model.is_fitted = True 
                 
            return model, None  # 直接返回加载的模型
        except FileNotFoundError:
            print(f"错误: 在 {model_path} 未找到 Pickle 文件")
            raise
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"警告: pickle 加载 {model_name.upper()} 模型失败: {e}")
            print("将回退到 PyTorch 加载流程，请确认模型文件类型与模型选择一致。")
            use_pickle = False
        except Exception as e:
            print(f"从 pickle 文件加载 {model_name.upper()} 模型时出错: {e}")
            raise
    # --- Pickle 处理结束 ---

    # --- 处理 PyTorch 模型 --- 
    model_class = models.get(model_name)
    if model_class is None:
        # 如果到这里 model_name 还不是 PyTorch 模型类型，则确实未知
        raise ValueError(f"未知的 PyTorch 模型类型: {model_name}")
    
    # 尝试加载 PyTorch 模型状态
    try:
        raw_checkpoint = _torch_load_compat(model_path, map_location=device)
        print(f"成功从以下路径加载 PyTorch 模型状态: {model_path}")
    except FileNotFoundError:
        print(f"错误: 在 {model_path} 未找到 PyTorch 模型文件")
        raise
    except RuntimeError as e:
        if "Invalid magic number" in str(e):
            print(f"错误: 尝试使用 torch.load 加载非 PyTorch 文件 ({model_path})。")
            print("如果这是一个 OCSVM 模型，它应该是一个 .pkl 文件，并需要不同的加载方式。")
        else:
            print(f"加载 PyTorch 模型状态时出错: {e}")
        raise
    except Exception as e:
        print(f"加载 PyTorch 模型时发生意外错误: {e}")
        raise

    checkpoint_meta = {}
    if isinstance(raw_checkpoint, dict) and any(
        key in raw_checkpoint for key in ("model_state", "state_dict", "input_channels")
    ):
        state_dict = raw_checkpoint.get("model_state") or raw_checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = {
                k: v for k, v in raw_checkpoint.items() if isinstance(v, torch.Tensor)
            }
        checkpoint_meta = {
            "input_channels": raw_checkpoint.get("input_channels"),
            "input_height": raw_checkpoint.get("input_height"),
            "input_width": raw_checkpoint.get("input_width"),
        }
    else:
        state_dict = raw_checkpoint

    def _shape_tuple(shape_value):
        if shape_value is None:
            return (None, None, None)
        if isinstance(shape_value, torch.Size):
            shape_value = tuple(int(v) for v in shape_value)
        if isinstance(shape_value, (list, tuple)):
            vals = list(shape_value)
            if len(vals) != 3:
                raise ValueError(f"期望输入形状为3个元素 (channels, height, width)，但得到 {vals}")
            normalized = []
            for v in vals:
                normalized.append(None if v is None else int(v))
            return tuple(normalized)
        if isinstance(shape_value, dict):
            return tuple(
                None if shape_value.get(k) is None else int(shape_value.get(k))
                for k in ("input_channels", "input_height", "input_width")
            )
        return (None, None, None)

    ckpt_shape = _shape_tuple(
        (
            checkpoint_meta.get("input_channels"),
            checkpoint_meta.get("input_height"),
            checkpoint_meta.get("input_width"),
        )
    )
    data_shape_tuple = _shape_tuple(data_shape)

    def _infer_state_shape(state_dict_obj):
        if not isinstance(state_dict_obj, dict):
            return (None, None, None), {}
        conv1 = state_dict_obj.get("conv1.weight")
        conv3 = state_dict_obj.get("conv3.weight")
        projection = state_dict_obj.get("projection.weight")
        inferred = [None, None, None]
        arch_kwargs = {}
        if conv1 is not None:
            inferred[0] = int(conv1.shape[1])
        if projection is not None:
            arch_kwargs["projection_input_dim"] = int(projection.shape[1])
            arch_kwargs["projection_output_dim"] = int(projection.shape[0])
        if conv1 is not None:
            arch_kwargs.setdefault("conv_channels", [])
            arch_kwargs["conv_channels"].append(int(conv1.shape[0]))
        if state_dict_obj.get("conv2.weight") is not None:
            arch_kwargs.setdefault("conv_channels", arch_kwargs.get("conv_channels", []))
            arch_kwargs["conv_channels"].append(int(state_dict_obj["conv2.weight"].shape[0]))
        if state_dict_obj.get("conv3.weight") is not None:
            arch_kwargs.setdefault("conv_channels", arch_kwargs.get("conv_channels", []))
            arch_kwargs["conv_channels"].append(int(state_dict_obj["conv3.weight"].shape[0]))
        if state_dict_obj.get("fc1.weight") is not None:
            arch_kwargs["fc_features"] = [
                int(state_dict_obj["fc1.weight"].shape[0]),
                int(state_dict_obj.get("fc2.weight").shape[0]) if state_dict_obj.get("fc2.weight") is not None else 100,
            ]
        if conv3 is not None and projection is not None:
            feature_map = int((projection.shape[1] / conv3.shape[0]) ** 0.5)
            inferred_hw = feature_map * 8
            inferred[1] = inferred_hw
            inferred[2] = inferred_hw
        return tuple(inferred), arch_kwargs

    state_shape_tuple, inferred_arch = _infer_state_shape(state_dict)

    shape_sources = []
    if any(v is not None for v in ckpt_shape):
        shape_sources.append(("checkpoint 元数据", ckpt_shape))
    if any(v is not None for v in data_shape_tuple):
        shape_sources.append(("数据样本", data_shape_tuple))
    if any(v is not None for v in state_shape_tuple):
        shape_sources.append(("模型权重推断", state_shape_tuple))

    resolved = [None, None, None]
    names = ("channels", "height", "width")
    for source_name, shape_vals in shape_sources:
        for idx, axis_name in enumerate(names):
            val = shape_vals[idx]
            if val is None:
                continue
            if resolved[idx] is not None and resolved[idx] != val:
                # 当权重推断出的尺寸与 checkpoint/数据 冲突时，以 checkpoint/数据 为准，忽略权重推断
                if source_name == "模型权重推断":
                    continue
                else:
                    raise ValueError(
                        f"{source_name} 指定的输入{axis_name}为 {val}，与其它来源的 {resolved[idx]} 不一致，请检查模型与数据是否匹配。"
                    )
            resolved[idx] = val

    resolved_shape = tuple(resolved)

    shape_str = "×".join(
        str(v) if v is not None else "?" for v in resolved_shape
    )
    print(f"模型输入形状约束: {shape_str}")

    cnn_like_models = {"cnn", "pucnn"}
    cnnt_like_models = {"cnnt", "pucnnt"}
    cntt_like_models = {"cntt", "pucnntransformer"}
    image_model_names = cnn_like_models | cnnt_like_models | cntt_like_models

    # 检查是否需要调整模型结构 (这部分逻辑可能只适用于 PyTorch 模型)
    if model_name in image_model_names:
        # 为CNNTransformer模型特别处理
        if model_name in cnnt_like_models and "projection.weight" in state_dict:
            # 直接从projection.weight的形状推断特征维度
            projection_shape = state_dict["projection.weight"].shape
            feature_dim = projection_shape[1]  # 512或2048等
            
            print(f"从投影层权重 {projection_shape} 推断特征维度: {feature_dim}")
            if resolved_shape[1] is not None and resolved_shape[2] is not None:
                print(
                    f"将使用 checkpoint/数据指定的输入尺寸: {resolved_shape[1]}x{resolved_shape[2]}"
                )
        # 其他CNN模型的原有逻辑
        elif "fc1.weight" in state_dict:
            fc1_shape = state_dict["fc1.weight"].shape
            # 计算输入特征维度
            feature_dim = fc1_shape[1]
            
            # 计算原始输入图像尺寸
            if model_name in cnn_like_models:
                # CNN模型中，feature_dim = 128 * final_size * final_size
                # 假设使用了3次池化，原始尺寸 = final_size * 2^3
                final_size = int((feature_dim / 128) ** 0.5)
                orig_img_size = final_size * 8  # 2^3 = 8
            else:  # cnnt
                # CNNTransformer模型中，feature_dim = 128 * final_size * final_size
                final_size = int((feature_dim / 128) ** 0.5)
                orig_img_size = final_size * 8
            
            # 计算原始输入维度
            orig_input_dim = 11 * orig_img_size * orig_img_size
            
            print(f"检测到模型原始输入尺寸: {orig_img_size}x{orig_img_size}")
            print(f"当前请求的输入尺寸: {int((input_dim/11)**0.5)}x{int((input_dim/11)**0.5)}")
            
            # 如果输入维度不匹配，优先使用请求的输入维度（实际数据的维度）
            if orig_input_dim != input_dim:
                print(f"警告: 模型原始维度与实际数据维度不匹配")
                print(f"将使用实际数据维度: {input_dim} 而不是原始模型维度: {orig_input_dim}")
                # 保持input_dim不变，使用实际数据的维度
    
    # 创建模型（若模型支持则传入形状参数）
    ctor_kwargs = {}
    can_provide_shape = all(val is not None for val in resolved_shape)

    if model_name in (cnn_like_models | cntt_like_models) and can_provide_shape:
        # CNN / CNN-TokenTransformer 接受 input_shape 参数
        ctor_kwargs["input_shape"] = resolved_shape
    else:
        if can_provide_shape:
            ctor_kwargs.update(
                {
                    "input_channels": resolved_shape[0],
                    "input_height": resolved_shape[1],
                    "input_width": resolved_shape[2],
                }
            )
        if model_name in image_model_names:
            ctor_kwargs.update(
                {
                    k: v
                    for k, v in inferred_arch.items()
                    if k in {"projection_input_dim", "projection_output_dim", "conv_channels", "fc_features"}
                }
            )

    try:
        model = model_class(prior, input_dim, **ctor_kwargs).to(device)
    except TypeError:
        # 对不接受可选参数的模型回退至旧行为
        model = model_class(prior, input_dim).to(device)

    # 加载状态
    try:
        model.load_state_dict(state_dict)
        print("成功加载模型权重")
    except RuntimeError as e:
        print(f"警告: 加载模型状态失败: {e}")
        print("将创建新模型并从头开始训练")
    
    model.eval()
    return model, resolved_shape

def preprocess_h5_data(data_path, model_dir, norm_params_path=None):
    """预处理 H5 格式的输入数据"""
    with h5py.File(data_path, 'r') as f:
        windows = f['windows'][:]
        positions = f['positions'][:]
    
    # 转换数据维度 (与训练数据保持一致)
    x = np.transpose(windows, (3, 2, 0, 1))  # (N, 11, 16, 16)
    
    # 优先使用指定的标准化参数文件
    if norm_params_path and os.path.exists(norm_params_path):
        print(f"加载标准化参数文件: {norm_params_path}")
        norm_params = _torch_load_compat(norm_params_path, map_location="cpu")
        mean_per_channel = norm_params['mean']
        std_per_channel = norm_params['std']
    else:
        # 尝试从模型目录加载
        default_norm_params_path = os.path.join(model_dir, 'normalization_params.pth')
        if os.path.exists(default_norm_params_path):
            print(f"加载默认标准化参数文件: {default_norm_params_path}")
            norm_params = _torch_load_compat(default_norm_params_path, map_location="cpu")
            mean_per_channel = norm_params['mean']
            std_per_channel = norm_params['std']
        else:
            print("警告：未找到标准化参数文件，使用默认值")
            n_channels = x.shape[1]
            mean_per_channel = [0.0] * n_channels
            std_per_channel = [1.0] * n_channels
    
    # 标准化数据
    for channel in range(len(mean_per_channel)):
        x[:, channel, :, :] = (x[:, channel, :, :] - mean_per_channel[channel]) / (std_per_channel[channel] + 1e-8)
    
    # 转换为 PyTorch 张量
    x = torch.FloatTensor(x)
    return x, positions

def predict_single(model, data, device):
    """对单个数据进行预测"""
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        prediction = torch.sign(output).item()
        confidence = torch.sigmoid(output).item()
        
        label = "Positive" if prediction > 0 else "Negative"
        return label, confidence

def calculate_gdr(
    positions,
    confidences,
    deposit_file="deposit.xlsx",
    threshold_distance=4.0,
    confidence_threshold=0.5,
    coord_offset=(0.0, 0.0),
):
    """Calculate GDR using prediction positions and deposit coordinates."""
    try:
        deposits_df = pd.read_excel(deposit_file)
        print(f"成功读取矿点文件: {deposit_file}")
        print(f"矿点数据列名: {deposits_df.columns.tolist()}")
        print(f"矿点数据形状: {deposits_df.shape}")

        possible_x_cols = ['X', 'x', '经度', 'longitude', 'Longitude', 'LONGITUDE']
        possible_y_cols = ['Y', 'y', '纬度', 'latitude', 'Latitude', 'LATITUDE']
        x_col = next((col for col in possible_x_cols if col in deposits_df.columns), None)
        y_col = next((col for col in possible_y_cols if col in deposits_df.columns), None)
        if x_col is None or y_col is None:
            if len(deposits_df.columns) >= 2:
                x_col = deposits_df.columns[0]
                y_col = deposits_df.columns[1]
            else:
                return 0.0, 0, 0, None

        deposit_coords = deposits_df[[x_col, y_col]].values
        total_deposits = len(deposit_coords)
        offset_x, offset_y = _normalize_coord_offset(coord_offset)

        high_confidence_positions = []
        for i, conf in enumerate(confidences):
            if conf > confidence_threshold:
                pos = positions[i].cpu().numpy() if isinstance(positions[i], torch.Tensor) else positions[i]
                pos = np.asarray(pos, dtype=np.float64)
                high_confidence_positions.append(np.array([pos[0] + offset_x, pos[1] + offset_y], dtype=np.float64))

        print(f"置信度>{confidence_threshold}的网格数量: {len(high_confidence_positions)}")
        if len(high_confidence_positions) == 0:
            return 0.0, 0, total_deposits, None

        high_confidence_positions = np.asarray(high_confidence_positions, dtype=np.float64)
        hit_deposits = 0
        hit_status = []
        min_distances = []
        hit_details = []

        for i, (deposit_x, deposit_y) in enumerate(deposit_coords):
            distances = np.sqrt(
                (high_confidence_positions[:, 0] - deposit_x) ** 2 +
                (high_confidence_positions[:, 1] - deposit_y) ** 2
            )
            min_distance = np.min(distances) if len(distances) > 0 else float('inf')
            min_distances.append(min_distance)
            if min_distance <= threshold_distance:
                hit_deposits += 1
                hit_status.append(True)
                hit_details.append(f"矿点{i+1} ({deposit_x:.1f}, {deposit_y:.1f}): 命中 (最近距离 {min_distance:.1f}m)")
            else:
                hit_status.append(False)
                hit_details.append(f"矿点{i+1} ({deposit_x:.1f}, {deposit_y:.1f}): 未命中 (最近距离 {min_distance:.1f}m)")

        print("\n矿点命中详情（前10个）:")
        for detail in hit_details[:10]:
            print(f"  {detail}")
        if len(hit_details) > 10:
            print(f"  ... 还有 {len(hit_details)-10} 个矿点")

        gdr = hit_deposits / total_deposits if total_deposits > 0 else 0.0
        plot_info = {
            'deposit_coords': deposit_coords,
            'hit_status': hit_status,
            'min_distances': min_distances,
            'high_confidence_positions': high_confidence_positions,
            'threshold_distance': threshold_distance,
            'confidence_threshold': confidence_threshold,
        }
        return gdr, hit_deposits, total_deposits, plot_info
    except FileNotFoundError:
        print(f"错误: 找不到矿点文件 {deposit_file}")
        return 0.0, 0, 0, None
    except Exception as e:
        print(f"计算GDR时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0, 0, None


def plot_deposit_distribution(plot_info, model_dir, model_type, all_positions, coord_offset=(0.0, 0.0)):
    """绘制矿点分布图，用红绿色区分被命中和未被命中的矿点"""
    if plot_info is None:
        print("无法绘制命中情况图：缺少绘图信息")
        return

    try:
        deposit_coords = plot_info['deposit_coords']
        hit_status = plot_info['hit_status']
        min_distances = plot_info['min_distances']
        high_confidence_positions = plot_info['high_confidence_positions']
        threshold_distance = plot_info['threshold_distance']
        confidence_threshold = plot_info['confidence_threshold']

        hit_coords = [coord for coord, hit in zip(deposit_coords, hit_status) if hit]
        miss_coords = [coord for coord, hit in zip(deposit_coords, hit_status) if not hit]
        offset_x, offset_y = _normalize_coord_offset(coord_offset)

        if hit_coords:
            hit_coords = np.array(hit_coords)

        if miss_coords:
            miss_coords = np.array(miss_coords)

        all_prediction_coords = []
        for pos in all_positions:
            coord = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
            coord = np.asarray(coord, dtype=np.float64)
            all_prediction_coords.append(np.array([coord[0] + offset_x, coord[1] + offset_y], dtype=np.float64))

        all_prediction_coords = np.array(all_prediction_coords)
        all_deposit_coords = np.asarray(deposit_coords, dtype=np.float64)
        combined_coords = np.vstack([all_prediction_coords, all_deposit_coords]) if len(all_prediction_coords) > 0 else all_deposit_coords
        x_min, x_max = np.min(combined_coords[:, 0]), np.max(combined_coords[:, 0])
        y_min, y_max = np.min(combined_coords[:, 1]), np.max(combined_coords[:, 1])

        plt.figure(figsize=(12, 10))

        if len(hit_coords) > 0:
            plt.scatter(hit_coords[:, 0], hit_coords[:, 1], c='green', s=100, alpha=0.8, marker='o', label=f'被命中矿点({len(hit_coords)}个)', edgecolors='darkgreen', linewidth=2)

        if len(miss_coords) > 0:
            plt.scatter(miss_coords[:, 0], miss_coords[:, 1], c='red', s=100, alpha=0.8, marker='x', label=f'未命中矿点({len(miss_coords)}个)', linewidth=3)

        if len(high_confidence_positions) > 0:
            plt.scatter(high_confidence_positions[:, 0], high_confidence_positions[:, 1], c='blue', s=10, alpha=0.3, marker='.', label=f'异常网格 (置信度>{confidence_threshold})')

        for i, (coord, hit, min_dist) in enumerate(zip(deposit_coords, hit_status, min_distances)):
            x, y = coord
            color = 'green' if hit else 'red'
            plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, color=color, weight='bold')

        print(f"预测数据集坐标范围: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})")
        print(f"X轴跨度: {x_max - x_min:.1f}, Y轴跨度: {y_max - y_min:.1f}")

        x_margin = (x_max - x_min) * 0.02
        y_margin = (y_max - y_min) * 0.02
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        print(f"设置的坐标轴范围: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})")

        plt.xlabel('X坐标', fontsize=12)
        plt.ylabel('Y坐标', fontsize=12)
        plt.title(f'矿点分布图 - {model_type.upper()}模型\n距离阈值: {threshold_distance}m, 置信度阈值: {confidence_threshold}', fontsize=14, weight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_filename = '矿点命中情况.png'
        plot_path = os.path.join(model_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"矿点命中情况图已保存到: {plot_path}")
        plt.close()

        hit_count = sum(hit_status)
        total_count = len(hit_status)
        gdr = hit_count / total_count if total_count > 0 else 0

        print(f"\n命中情况统计:")
        print(f"  总矿点数: {total_count}")
        print(f"  被命中矿点数: {hit_count} (绿色圆点)")
        print(f"  未命中矿点数: {total_count - hit_count} (红色叉号)")
        print(f"  GDR: {gdr:.2%}")
        print(f"  异常网格数: {len(high_confidence_positions)}")
    except Exception as e:
        print(f"绘制矿点命中情况图时发生错误: {e}")
        import traceback
        traceback.print_exc()


def save_predictions(
    positions,
    predictions,
    confidences,
    labels,
    output_file,
    coord_offset=(0.0, 0.0),
    metadata=None,
    mineral_points=None,
):
    """保存预测结果为h5格式"""
    predictions_numeric = np.array([1 if p == 'Positive' else -1 for p in predictions], dtype=np.int8)
    offset_x, offset_y = _normalize_coord_offset(coord_offset)

    try:
        if isinstance(positions[0], torch.Tensor):
            positions = [pos.cpu().numpy() for pos in positions]

        positions_adjusted = np.vstack(positions).astype(np.float32)
        positions_adjusted[:, 0] += offset_x
        positions_adjusted[:, 1] += offset_y
    except Exception as e:
        print(f"转换positions时出错: {e}")
        print(f"positions的第一个元素: {positions[0]}")
        print(f"positions的形状: {len(positions)}")
        raise

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('positions', data=positions_adjusted)
        f.create_dataset('predictions', data=predictions_numeric)
        f.create_dataset('confidences', data=np.array(confidences, dtype=np.float32))
        f.create_dataset('labels', data=np.array(labels, dtype=np.int8))
        if mineral_points is not None:
            f.create_dataset('mineral_points', data=np.asarray(mineral_points, dtype=np.float32))

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, np.integer, np.floating, str, bytes)):
                    f.attrs[key] = value

        f.attrs['total_predictions'] = len(positions)
        f.attrs['positive_count'] = sum(1 for p in predictions if p == 'Positive')
        f.attrs['negative_count'] = sum(1 for p in predictions if p == 'Negative')
        f.attrs['true_positive_count'] = sum(1 for l in labels if l == 1)
        f.attrs['true_negative_count'] = sum(1 for l in labels if l == 0)


def predict_batch(model, data, device):
    """对一批数据进行预测"""
    if hasattr(model, "predict") and hasattr(model, "predict_proba") and getattr(model, "is_fitted", False):
        probabilities = np.asarray(model.predict_proba(data), dtype=np.float32)
        positive_confidence = probabilities[:, 1].reshape(-1)
        predictions = np.asarray(model.predict(data)).reshape(-1)
        return ["Positive" if p > 0 else "Negative" for p in predictions], positive_confidence

    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)
        # 使用torch.sign来确定预测类别
        predictions = torch.sign(outputs).cpu().numpy()
        confidences = torch.sigmoid(outputs).cpu().numpy()
        # 将预测结果转换为字符串标签
        return ["Positive" if p > 0 else "Negative" for p in predictions.flatten()], confidences.flatten()

class WindowDataset(Dataset):
    def __init__(self, data_file, label_file, model_dir, use_custom_norm=False, norm_params_path=None):
        # 加载数据
        with h5py.File(data_file, 'r') as f:
            self.windows = f['windows'][:].astype(np.float32)  # 确保使用float32类型
            self.positions = f['positions'][:].astype(np.float32)
        
        # 加载标签文件并生成标签
        with h5py.File(label_file, 'r') as f:
            label_windows = f['windows'][:].astype(np.float32)  # (32, 32, 1, N)
        
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
        resolved_norm_params_path = norm_params_path or os.path.join(model_dir, 'normalization_params.pth')
        if use_custom_norm and os.path.exists(resolved_norm_params_path):
            print(f"使用自定义标准化参数文件: {resolved_norm_params_path}")
            norm_params = _torch_load_compat(resolved_norm_params_path, map_location="cpu")
            self.means = [float(v) for v in norm_params['mean']]
            self.stds = [float(v) for v in norm_params['std']]
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
        
        # 转换为PyTorch张量并确保使用float32类型
        self.windows = torch.FloatTensor(self.windows)
        self.positions = torch.FloatTensor(self.positions)
        self.labels = torch.LongTensor(self.labels)
        
        # 打印数据集信息
        print(f"数据集大小: {len(self.windows)}")
        print(f"正样本数量: {torch.sum(self.labels == 1).item()}")
        print(f"负样本数量: {torch.sum(self.labels == -1).item()}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.positions[idx], self.labels[idx]


if False:
    main()

# 确保 main 函数中的默认参数也正确
def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--model-type', type=str, default='cnnt',
                       help='模型类型 (例如: linear, 3lp, mlp, cnn, cnnt, cntt, pucnn, pucnnt, pucnntransformer, rf, purf, ocsvm, 2step)')
    # 确认这个默认路径是正确的 .pkl 文件路径
    parser.add_argument('--model-path', type=str, default='result/cnnt-9-0.2-10%/model.pth',
                       help='训练好的模型路径 (.pth 对应神经网络模型, .pkl 对应 RF/OCSVM/2step/purf)')
    parser.add_argument('--input', type=str, default='data/windows_combined_data.h5',
                       help='输入数据文件 (h5)')
    parser.add_argument('--label-file', type=str, default='data/deposit_labels.h5',
                       help='标签文件 (h5)')
    # 确认这个先验概率与训练时使用的匹配
    parser.add_argument('--prior', type=float, default=0.2,
                       help='先验概率值')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    # 确认这个标准化参数路径正确
    parser.add_argument('--norm-params', type=str, default='result/cnnt-9-0.2-10%/normalization_params.pth',
                       help='标准化参数文件路径 (.pth)')
    parser.add_argument('--output-dir', type=str, default='predictions', 
                       help='预测结果输出目录')
    parser.add_argument('--img_size', default=16, type=int, help='输入图像的尺寸 (假设为正方形)')
    parser.add_argument('--patch-stride', default=None, type=int, help='预测切窗步长；默认等于 --img_size')
    parser.add_argument('--deposit-file', type=str, default='deposit.xlsx',
                       help='矿点坐标文件路径 (.xlsx)')
    parser.add_argument('--distance-threshold', type=float, default=4,
                       help='GDR计算的距离阈值(米)')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='GDR计算的置信度阈值')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 获取模型目录
        model_dir = os.path.dirname(args.model_path)

        # 加载数据
        prediction_stride = int(args.patch_stride or args.img_size)
        print(f"预测窗口大小: {args.img_size} | 预测步长: {prediction_stride}")
        dataset = PredictionDataset(
            args.input,
            args.label_file,
            model_dir,
            use_custom_norm=True,
            norm_params_path=args.norm_params,
            patch_size=args.img_size,
            patch_stride=prediction_stride,
            use_reflect_padding=True,
            selected_channels=None,
        )
        
        # 从数据中获取实际输入维度
        sample_data = dataset[0][0]  # 获取第一个样本
        input_channels = sample_data.shape[0]  # 通道数
        input_height = sample_data.shape[1]  # 高度
        input_width = sample_data.shape[2]  # 宽度
        actual_input_dim = input_channels * input_height * input_width
        coord_offset = getattr(dataset, "coord_offset", (input_width / 2.0, input_height / 2.0))
        
        print(f"检测到实际输入维度: {actual_input_dim} ({input_channels}×{input_height}×{input_width})")
        
        # 使用实际输入维度而不是固定尺寸
        input_dim = actual_input_dim
        model, _ = load_model(
            args.model_type,
            args.model_path,
            args.prior,
            input_dim,
            device,
            data_shape=(input_channels, input_height, input_width),
        )
        print(f"模型加载成功: {args.model_path}")
        
        # 加载数据
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print(f"数据加载成功: {args.input}")
        print(f"标签加载成功: {args.label_file}")
        print(f"数据集大小: {len(dataset)}")
        
        if args.norm_params:
            print(f"使用自定义标准化参数文件: {args.norm_params}")
        
        # 创建输出目录
        # predictions_dir = args.output_dir # 不再需要这个作为指标文件的目录
        # os.makedirs(predictions_dir, exist_ok=True)
        # output_file = os.path.join(predictions_dir, f'predictions_{args.model_type}.h5')
        
        # --- 恢复 h5 输出路径到 predictions 目录 --- 
        predictions_dir = args.output_dir # 使用命令行参数指定的输出目录
        os.makedirs(predictions_dir, exist_ok=True) # 确保 predictions 目录存在
        output_file = os.path.join(predictions_dir, f'predictions_{args.model_type}.h5') # <-- h5 文件路径
        
        # 进行预测
        all_positions = []
        all_predictions = []
        all_confidences = []
        all_labels = []
        
        print("开始预测...")
        for batch_idx, (data, positions, labels) in enumerate(dataloader):
            predictions, confidences = predict_batch(model, data, device)
            all_positions.extend(positions)
            all_predictions.extend(predictions)
            all_confidences.extend(confidences)
            all_labels.extend(labels)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"已处理 {batch_idx + 1} 批次...")
        
        # 计算召回率
        all_labels_np = torch.stack(all_labels).cpu().numpy()
        all_predictions_np = np.array([1 if p == 'Positive' else -1 for p in all_predictions])
        
        # 计算混淆矩阵指标
        true_positives = np.sum((all_labels_np == 1) & (all_predictions_np == 1))
        false_negatives = np.sum((all_labels_np == 1) & (all_predictions_np == -1))
        true_negatives = np.sum((all_labels_np == -1) & (all_predictions_np == -1))
        false_positives = np.sum((all_labels_np == -1) & (all_predictions_np == 1))
        
        # 计算GDR (Gold Deposit Hit Rate) 指标替代召回率
        print("\n正在计算GDR指标...")
        gdr, hit_deposits, total_deposits, plot_info = calculate_gdr(
            all_positions, 
            all_confidences, 
            args.deposit_file, 
            args.distance_threshold, 
            args.confidence_threshold,
            coord_offset=coord_offset,
        )
        
        # 不再计算精确率、F1分数和准确率
        
        # 计算 Pr[f(X)=1] 和 GDR/Pr[f(X)=1] 指标
        pr_positive = sum(1 for p in all_predictions if p == 'Positive') / len(all_predictions)
        gdr_over_pr = gdr / pr_positive if pr_positive > 0 else 0.0
        
        # 计算 AAR (Anomaly Area Rate) 指标
        # AAR = 异常区域率 = 预测为正类的区域比例 = (TP + FP) / 总样本数
        predicted_positive_ratio = (true_positives + false_positives) / len(all_labels_np)
        aar = predicted_positive_ratio  # AAR指标
        
        # 计算 PAR (Prediction Area Ratio) 指标
        # PAR = 预测区域积分比 = 置信度>0.5的网格数 / 总网格数
        confidences_above_threshold = sum(1 for conf in all_confidences if conf > 0.5)
        par = confidences_above_threshold / len(all_confidences)  # PAR指标
        
        # 计算 TARR (Target Area Reduction Ratio) 指标
        # TARR = (1 - AAR) × GDR
        area_reduction_ratio = 1 - aar  # 区域缩减比例
        tarr = area_reduction_ratio * gdr  # TARR指标
        
        # 计算 TARR-P (先验概率增强的TARR) 指标
        # TARR-P = (1 - AAR/p) × GDR
        # 其中: AAR = 异常区域率, p = 先验概率, GDR = 金矿击中率
        prior_probability = args.prior  # 从命令行参数获取先验概率
        if prior_probability > 0:
            aar_over_p = aar / prior_probability  # AAR/p比值
            prior_adjusted_reduction = 1 - aar_over_p  # 1 - AAR/p
            tarr_p = prior_adjusted_reduction * gdr  # TARR-P指标
        else:
            aar_over_p = float('inf')  # 先验概率为0时，比值为无穷大
            tarr_p = 0.0  # 先验概率为0时，TARR-P无意义
        
        # 保存预测结果
        save_predictions(
            all_positions,
            all_predictions,
            all_confidences,
            all_labels_np,
            output_file,
            coord_offset=coord_offset,
            metadata=getattr(dataset, "metadata", None),
            mineral_points=getattr(dataset, "mineral_points", None),
        ) # 确保传递 all_labels_np
        
        # 绘制矿点击中情况图
        print("\n正在绘制矿点击中情况图...")
        plot_deposit_distribution(plot_info, model_dir, args.model_type, all_positions, coord_offset=coord_offset)
        
        # 打印结果统计
        print(f"预测完成，结果已保存到: {output_file}")
        print(f"总预测点数: {len(all_positions)}")
        print(f"正类预测数: {sum(1 for p in all_predictions if p == 'Positive')}")
        print(f"负类预测数: {sum(1 for p in all_predictions if p == 'Negative')}")
        print(f"置信度>0.5的网格数: {confidences_above_threshold}")
        
        # 打印性能指标
        print("\n预测性能指标:")
        print(f"GDR (金矿击中率): {gdr:.4f} ({gdr*100:.2f}%) - 被成功识别的矿点比例 ({hit_deposits}/{total_deposits})")
        print(f"AAR (异常区域率): {aar:.4f} ({aar*100:.2f}%) - 预测为异常的区域比例")
        print(f"PAR (预测区域积分比): {par:.4f} ({par*100:.2f}%) - 置信度>0.5的区域比例")
        print(f"Pr[f(X)=1]: {pr_positive:.4f} - 预测为正类的概率")
        print(f"GDR/Pr[f(X)=1]: {gdr_over_pr:.4f} - GDR与预测为正概率的比值")
        print(f"TARR (目标区域缩减比): {tarr:.4f} - 区域缩减能力与找矿能力的综合指标")
        print(f"  └─ AAR (异常区域率): {aar:.4f} ({aar*100:.2f}%)")
        print(f"  └─ 区域缩减比例: {area_reduction_ratio:.4f} ({area_reduction_ratio*100:.2f}%)")
        print(f"TARR-P (先验概率增强): {tarr_p:.4f} - 考虑先验概率的区域缩减指标")
        print(f"  └─ 先验概率: {prior_probability:.4f} ({prior_probability*100:.2f}%)")
        if prior_probability > 0:
            print(f"  └─ AAR/p比值: {aar_over_p:.4f} - 异常区域率与先验概率的比值")
            if aar_over_p <= 1:
                print(f"  └─ 评价: 预测区域 ≤ 先验期望，区域缩减有效")
            else:
                print(f"  └─ 评价: 预测区域 > 先验期望，区域缩减不足")
        else:
            print(f"  └─ AAR/p比值: ∞ - 先验概率为0，TARR-P无意义")
        
        # 打印混淆矩阵
        print("\n混淆矩阵:")
        print(f"真正例 (TP): {true_positives} - 正确预测为正类的正样本")
        print(f"假负例 (FN): {false_negatives} - 错误预测为负类的正样本")
        print(f"真负例 (TN): {true_negatives} - 正确预测为负类的负样本")
        print(f"假正例 (FP): {false_positives} - 错误预测为正类的负样本")
        
        # --- 修改指标文件保存路径 --- 
        # 将评估指标写入文本文件
        # metrics_file = os.path.join(predictions_dir, f'metrics_{args.model_type}.txt') # 旧路径
        metrics_file = os.path.join(model_dir, f'metrics_{args.model_type}.txt') # <-- 新路径：使用 model_dir
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"预测日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型类型: {args.model_type}\n")
            f.write(f"模型路径: {args.model_path}\n")
            f.write(f"数据路径: {args.input}\n")
            f.write(f"矿点文件: {args.deposit_file}\n")
            f.write(f"距离阈值: {args.distance_threshold}m\n")
            f.write(f"置信度阈值: {args.confidence_threshold}\n")
            f.write(f"矿点分布图: 矿点击中情况.png\n")
            f.write(f"总预测点数: {len(all_positions)}\n")
            f.write(f"正类预测数: {sum(1 for p in all_predictions if p == 'Positive')}\n")
            f.write(f"负类预测数: {sum(1 for p in all_predictions if p == 'Negative')}\n")
            f.write(f"置信度>0.5的网格数: {confidences_above_threshold}\n\n")
            
            f.write("预测性能指标:\n")
            f.write(f"GDR (金矿击中率): {gdr:.4f} ({gdr*100:.2f}%)\n")
            f.write(f"击中矿点数: {hit_deposits}/{total_deposits}\n")
            f.write(f"AAR (异常区域率): {aar:.4f} ({aar*100:.2f}%)\n")
            f.write(f"PAR (预测区域积分比): {par:.4f} ({par*100:.2f}%)\n")
            f.write(f"Pr[f(X)=1]: {pr_positive:.4f}\n")
            f.write(f"GDR/Pr[f(X)=1]: {gdr_over_pr:.4f}\n")
            f.write(f"TARR (目标区域缩减比): {tarr:.4f}\n")
            f.write(f"AAR (异常区域率): {aar:.4f} ({aar*100:.2f}%)\n")
            f.write(f"区域缩减比例: {area_reduction_ratio:.4f} ({area_reduction_ratio*100:.2f}%)\n")
            f.write(f"TARR-P (先验概率增强): {tarr_p:.4f}\n")
            f.write(f"先验概率: {prior_probability:.4f} ({prior_probability*100:.2f}%)\n")
            if prior_probability > 0:
                f.write(f"AAR/p比值: {aar_over_p:.4f}\n")
                if aar_over_p <= 1:
                    f.write(f"评价: 预测区域 ≤ 先验期望，区域缩减有效\n")
                else:
                    f.write(f"评价: 预测区域 > 先验期望，区域缩减不足\n")
            else:
                f.write(f"AAR/p比值: ∞ - 先验概率为0，TARR-P无意义\n")
            
            f.write("\n")
            
            f.write("混淆矩阵:\n")
            f.write(f"真正例 (TP): {true_positives}\n")
            f.write(f"假负例 (FN): {false_negatives}\n")
            f.write(f"真负例 (TN): {true_negatives}\n")
            f.write(f"假正例 (FP): {false_positives}\n")
            f.write("\n")
        
        print(f"评估指标已保存到: {metrics_file}")
        
    except Exception as e:
        print(f"\n程序执行出错：")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()

if False:
    main() 


class PredictionDataset(Dataset):
    def __init__(
        self,
        data_file,
        label_file,
        model_dir,
        use_custom_norm=False,
        norm_params_path=None,
        patch_size=None,
        patch_stride=None,
        use_reflect_padding=False,
        selected_channels=None,
    ):
        patch_size = int(patch_size or 16)
        patch_stride = int(patch_stride or patch_size)

        self.windows, coordinates, metadata, feature_mode = _load_feature_tensor(
            data_file,
            patch_size,
            patch_stride,
            enable_reflect_padding=use_reflect_padding,
            selected_channels=selected_channels,
        )
        self.feature_mode = feature_mode
        self.metadata = metadata or {}
        self.mineral_points = None

        label_ext = os.path.splitext(label_file)[1].lower()
        if label_ext in {".txt", ".csv", ".tsv", ".dat"}:
            minerals = _read_mineral_points(label_file)
            self.mineral_points = minerals[["x", "y"]].to_numpy(dtype=np.float64)
            positive_mask = _window_contains_minerals(coordinates, minerals, self.metadata, patch_size) > 0
            labels = np.where(positive_mask, 1, -1).astype(np.int32)
        elif label_ext in {".h5", ".hdf5"}:
            with h5py.File(label_file, "r") as f:
                label_windows = None
                if "windows" in f:
                    label_windows = np.asarray(f["windows"][:], dtype=np.float32)
                elif "label_windows" in f:
                    label_windows = np.asarray(f["label_windows"][:], dtype=np.float32)
                elif "labels" in f:
                    labels = np.asarray(f["labels"][:]).reshape(-1)
                else:
                    raise ValueError("H5 标签文件必须包含 labels/windows/label_windows 数据集。")

                if label_windows is not None:
                    sample_count = len(coordinates) if coordinates is not None else label_windows.shape[-1]
                    sample_axis = next((idx for idx, dim in enumerate(label_windows.shape) if dim == sample_count), None)
                    if sample_axis is None:
                        sample_axis = label_windows.ndim - 1
                    label_windows = np.moveaxis(label_windows, sample_axis, 0)
                    if label_windows.ndim == 1:
                        labels = label_windows
                    else:
                        positive_mask = np.any(label_windows == 1, axis=tuple(range(1, label_windows.ndim)))
                        labels = np.where(positive_mask, 1, -1)
        else:
            raise ValueError("标签文件必须是 H5 或矿点 txt/dat/csv/tsv。")

        labels = np.asarray(labels).reshape(-1)
        if labels.size == 0:
            raise ValueError("标签文件未生成有效样本。")
        self.labels = np.where(labels > 0, 1, -1).astype(np.int32)

        if coordinates is None:
            coordinates = np.column_stack(
                [
                    np.arange(len(self.windows), dtype=np.float64),
                    np.zeros(len(self.windows), dtype=np.float64),
                ]
            )
        self.positions = np.asarray(_patch_indices_to_geo(coordinates, self.metadata), dtype=np.float32)
        if len(self.positions) != len(self.windows):
            raise ValueError(f"特征/坐标数量不一致: features={len(self.windows)}, positions={len(self.positions)}")

        resolved_norm_params_path = norm_params_path or os.path.join(model_dir, "normalization_params.pth")
        if use_custom_norm and os.path.exists(resolved_norm_params_path):
            print(f"使用自定义标准化参数文件: {resolved_norm_params_path}")
            norm_params = _torch_load_compat(resolved_norm_params_path, map_location="cpu")
            self.means = [float(v) for v in norm_params["mean"]]
            self.stds = [float(v) for v in norm_params["std"]]
        else:
            self.means = []
            self.stds = []
            for i in range(self.windows.shape[1]):
                self.means.append(float(np.mean(self.windows[:, i])))
                self.stds.append(float(np.std(self.windows[:, i])))

        for i in range(self.windows.shape[1]):
            self.windows[:, i] = (self.windows[:, i] - self.means[i]) / (self.stds[i] + 1e-8)

        self.windows = torch.FloatTensor(self.windows)
        self.positions = torch.FloatTensor(self.positions)
        self.labels = torch.LongTensor(self.labels)
        self.coord_offset = (0.0, 0.0)

        print(f"数据集大小: {len(self.windows)}")
        print(describe_selected_channels(self.metadata.get("selected_channel_indices"), self.metadata.get("available_channel_names", [])))
        print(f"正样本数: {torch.sum(self.labels == 1).item()}")
        print(f"负样本数: {torch.sum(self.labels == -1).item()}")
        print(f"输入模式: {self.feature_mode}")
        print(f"预测补边: {'reflect' if use_reflect_padding and self.feature_mode == 'patches' else 'none'}")
        print(f"标签模式: {'coordinates' if label_ext in {'.txt', '.csv', '.tsv', '.dat'} else 'h5'}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.positions[idx], self.labels[idx]


if __name__ == '__main__':
    main()
