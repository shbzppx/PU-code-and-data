import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import (LinearClassifier, ThreeLayerPerceptron, MultiLayerPerceptron,
                   CNN, CNNTransformer, CNNTokenTransformer, RandomForestBinaryClassifier,
                   OneClassSVMClassifier, PURandomForestClassifier, TwoStepPULearning)
import h5py
import os
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

def load_model(model_name, model_path, prior, input_dim, device, data_shape=None):
    """加载训练好的模型"""
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
        # OCSVM / 2step / PU-Random Forest 将单独处理
    }
    
    # --- 处理使用 pickle 保存的模型 --- 
    if model_name in {"rf", "ocsvm", "2step", "purf"}: # <-- 合并条件
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"成功从 pickle 文件加载 {model_name.upper()} 模型: {model_path}")
            
            # 可选：进行类型检查
            if model_name == "ocsvm" and not isinstance(model, OneClassSVMClassifier):
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
                 
            return model # 直接返回加载的模型
        except FileNotFoundError:
            print(f"错误: 在 {model_path} 未找到 Pickle 文件")
            raise
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
        saved_state = torch.load(model_path, map_location=device)
        if isinstance(saved_state, dict) and any(key in saved_state for key in ("model_state", "state_dict")):
            saved_state = saved_state.get("model_state") or saved_state.get("state_dict")
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

    # 检查是否需要调整模型结构 (这部分逻辑可能只适用于 PyTorch 模型)
    cnn_like_models = {"cnn", "pucnn"}
    cnnt_like_models = {"cnnt", "pucnnt"}
    cntt_like_models = {"cntt", "pucnntransformer"}
    image_model_names = cnn_like_models | cnnt_like_models | cntt_like_models

    if model_name in image_model_names:
        # 为CNNTransformer模型特别处理
        if model_name in cnnt_like_models and "projection.weight" in saved_state:
            # 直接从projection.weight的形状推断特征维度
            projection_shape = saved_state["projection.weight"].shape
            feature_dim = projection_shape[1]  # 512或2048等
            
            print(f"从投影层权重 {projection_shape} 推断特征维度: {feature_dim}")
            print(f"当前请求的输入尺寸: {int((input_dim/11)**0.5)}x{int((input_dim/11)**0.5)}")
            
            # 我们将继续使用请求的输入维度，让模型适应它
            print(f"将使用请求的输入维度: {input_dim}")
        # 其他CNN模型的原有逻辑
        elif "fc1.weight" in saved_state:
            fc1_shape = saved_state["fc1.weight"].shape
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
    
    # 创建模型
    ctor_kwargs = {}
    if data_shape is not None and model_name in image_model_names:
        if model_name in cnn_like_models or model_name in cntt_like_models:
            ctor_kwargs["input_shape"] = data_shape
        elif model_name in cnnt_like_models:
            ctor_kwargs["input_channels"] = data_shape[0]
    try:
        model = model_class(prior, input_dim, **ctor_kwargs).to(device)
    except TypeError:
        model = model_class(prior, input_dim).to(device)
    
    # 加载状态
    try:
        model.load_state_dict(saved_state)
        print("成功加载模型权重")
    except RuntimeError as e:
        print(f"警告: 加载模型状态失败: {e}")
        print("将创建新模型并从头开始训练")
    
    model.eval()
    return model

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
        norm_params = torch.load(norm_params_path)
        mean_per_channel = norm_params['mean']
        std_per_channel = norm_params['std']
    else:
        # 尝试从模型目录加载
        default_norm_params_path = os.path.join(model_dir, 'normalization_params.pth')
        if os.path.exists(default_norm_params_path):
            print(f"加载默认标准化参数文件: {default_norm_params_path}")
            norm_params = torch.load(default_norm_params_path)
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

def calculate_gdr(positions, confidences, deposit_file="deposit.xlsx", threshold_distance=4.0, confidence_threshold=0.5):
    """
    计算GDR (Gold Deposit Hit Rate) 指标
    
    Args:
        positions: 预测网格的位置坐标 [(x, y), ...]
        confidences: 对应的置信度 [conf1, conf2, ...]
        deposit_file: 矿点坐标文件路径
        threshold_distance: 距离阈值(米)，默认200m
        confidence_threshold: 置信度阈值，默认0.5
    
    Returns:
        gdr: GDR指标值 (0-1)
        hit_deposits: 被击中的矿点数量
        total_deposits: 总矿点数量
    """
    try:
        # 读取矿点坐标
        deposits_df = pd.read_excel(deposit_file)
        print(f"成功读取矿点文件: {deposit_file}")
        print(f"矿点数据列名: {deposits_df.columns.tolist()}")
        print(f"矿点数据形状: {deposits_df.shape}")
        
        # 假设矿点坐标列名为 'X', 'Y' 或 'x', 'y' 或其他可能的变体
        possible_x_cols = ['X', 'x', '经度', 'longitude', 'Longitude', 'LONGITUDE']
        possible_y_cols = ['Y', 'y', '纬度', 'latitude', 'Latitude', 'LATITUDE']
        
        x_col = None
        y_col = None
        
        for col in possible_x_cols:
            if col in deposits_df.columns:
                x_col = col
                break
        
        for col in possible_y_cols:
            if col in deposits_df.columns:
                y_col = col
                break
        
        if x_col is None or y_col is None:
            print(f"警告: 无法识别矿点坐标列名，请检查文件格式")
            print(f"可用列名: {deposits_df.columns.tolist()}")
            # 如果无法识别列名，尝试使用前两列
            if len(deposits_df.columns) >= 2:
                x_col = deposits_df.columns[0]
                y_col = deposits_df.columns[1]
                print(f"尝试使用前两列作为坐标: X={x_col}, Y={y_col}")
            else:
                return 0.0, 0, 0
        
        deposit_coords = deposits_df[[x_col, y_col]].values
        total_deposits = len(deposit_coords)
        print(f"总矿点数量: {total_deposits}")
        print(f"前5个矿点坐标: {deposit_coords[:5]}")
        
        # 获取置信度大于阈值的网格位置
        high_confidence_positions = []
        for i, conf in enumerate(confidences):
            if conf > confidence_threshold:
                if isinstance(positions[i], torch.Tensor):
                    pos = positions[i].cpu().numpy()
                else:
                    pos = positions[i]
                # 减去偏移量（如果需要）
                pos_adjusted = pos - 8  # 根据代码中的偏移量调整
                high_confidence_positions.append(pos_adjusted)
        
        print(f"置信度>{confidence_threshold}的网格数量: {len(high_confidence_positions)}")
        
        if len(high_confidence_positions) == 0:
            return 0.0, 0, total_deposits
        
        high_confidence_positions = np.array(high_confidence_positions)
        
        # 计算每个矿点是否被击中
        hit_deposits = 0
        hit_details = []
        hit_status = []  # 记录每个矿点的击中状态
        min_distances = []  # 记录每个矿点的最近距离
        
        for i, deposit_coord in enumerate(deposit_coords):
            deposit_x, deposit_y = deposit_coord
            
            # 计算该矿点到所有高置信度网格的距离
            distances = np.sqrt((high_confidence_positions[:, 0] - deposit_x)**2 + 
                              (high_confidence_positions[:, 1] - deposit_y)**2)
            
            # 检查是否有网格在阈值距离内
            min_distance = np.min(distances) if len(distances) > 0 else float('inf')
            min_distances.append(min_distance)
            
            if min_distance <= threshold_distance:
                hit_deposits += 1
                hit_status.append(True)
                hit_details.append(f"矿点{i+1} ({deposit_x:.1f}, {deposit_y:.1f}): 击中 (最近距离: {min_distance:.1f}m)")
            else:
                hit_status.append(False)
                hit_details.append(f"矿点{i+1} ({deposit_x:.1f}, {deposit_y:.1f}): 未击中 (最近距离: {min_distance:.1f}m)")
        
        # 打印详细信息（前10个）
        print("\n矿点击中详情（前10个）:")
        for detail in hit_details[:10]:
            print(f"  {detail}")
        if len(hit_details) > 10:
            print(f"  ... 还有 {len(hit_details)-10} 个矿点")
        
        gdr = hit_deposits / total_deposits if total_deposits > 0 else 0.0
        
        # 返回绘图所需的额外信息
        plot_info = {
            'deposit_coords': deposit_coords,
            'hit_status': hit_status,
            'min_distances': min_distances,
            'high_confidence_positions': high_confidence_positions,
            'threshold_distance': threshold_distance,
            'confidence_threshold': confidence_threshold
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

def plot_deposit_distribution(plot_info, model_dir, model_type, all_positions):
    """
    绘制矿点分布图，用红绿颜色区分被击中和未被击中的矿点
    
    Args:
        plot_info: 包含绘图信息的字典
        model_dir: 模型目录路径
        model_type: 模型类型
        all_positions: 预测数据集中所有网格的坐标
    """
    if plot_info is None:
        print("无法绘制击中情况图：缺少绘图信息")
        return
    
    try:
        deposit_coords = plot_info['deposit_coords']
        hit_status = plot_info['hit_status']
        min_distances = plot_info['min_distances']
        high_confidence_positions = plot_info['high_confidence_positions']
        threshold_distance = plot_info['threshold_distance']
        confidence_threshold = plot_info['confidence_threshold']
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 分离被击中和未被击中的矿点
        hit_coords = [coord for coord, hit in zip(deposit_coords, hit_status) if hit]
        miss_coords = [coord for coord, hit in zip(deposit_coords, hit_status) if not hit]
        
        # 绘制矿点
        if hit_coords:
            hit_coords = np.array(hit_coords)
            plt.scatter(hit_coords[:, 0], hit_coords[:, 1], 
                       c='green', s=100, alpha=0.8, marker='o', 
                       label=f'被击中矿点 ({len(hit_coords)}个)', edgecolors='darkgreen', linewidth=2)
        
        if miss_coords:
            miss_coords = np.array(miss_coords)
            plt.scatter(miss_coords[:, 0], miss_coords[:, 1], 
                       c='red', s=100, alpha=0.8, marker='x', 
                       label=f'未击中矿点 ({len(miss_coords)}个)', linewidth=3)
        
        # 绘制高置信度网格位置（可选，用小点显示）
        if len(high_confidence_positions) > 0:
            plt.scatter(high_confidence_positions[:, 0], high_confidence_positions[:, 1], 
                       c='blue', s=10, alpha=0.3, marker='.', 
                       label=f'异常网格 (置信度>{confidence_threshold})')
        
        # 添加矿点编号
        for i, (coord, hit, min_dist) in enumerate(zip(deposit_coords, hit_status, min_distances)):
            x, y = coord
            color = 'green' if hit else 'red'
            plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color=color, weight='bold')
        
        # 基于预测数据集中所有网格的坐标来计算坐标轴范围
        all_prediction_coords = []
        for pos in all_positions:
            if isinstance(pos, torch.Tensor):
                coord = pos.cpu().numpy()
            else:
                coord = pos
            # 应用相同的偏移量调整
            coord_adjusted = coord - 8
            all_prediction_coords.append(coord_adjusted)
        
        all_prediction_coords = np.array(all_prediction_coords)
        
        # 计算预测数据集的坐标范围
        x_min, x_max = np.min(all_prediction_coords[:, 0]), np.max(all_prediction_coords[:, 0])
        y_min, y_max = np.min(all_prediction_coords[:, 1]), np.max(all_prediction_coords[:, 1])
        
        print(f"预测数据集坐标范围: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})")
        print(f"X轴跨度: {x_max - x_min:.1f}, Y轴跨度: {y_max - y_min:.1f}")
        
        # 添加较小的边距（2%）
        x_margin = (x_max - x_min) * 0.02
        y_margin = (y_max - y_min) * 0.02
        
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin
        
        # 设置坐标轴范围（严格按照数据范围）
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        print(f"设置的坐标轴范围: X({x_min:.1f}, {x_max:.1f}), Y({y_min:.1f}, {y_max:.1f})")
        
        # 设置图形属性
        plt.xlabel('X坐标', fontsize=12)
        plt.ylabel('Y坐标', fontsize=12)
        plt.title(f'矿点分布图 - {model_type.upper()}模型\n'
                 f'距离阈值: {threshold_distance}m, 置信度阈值: {confidence_threshold}', 
                 fontsize=14, weight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 不使用equal比例，避免自动扩大坐标轴范围
        # plt.axis('equal')  # 注释掉这行，避免强制等比例导致范围扩大
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plot_filename = '矿点击中情况.png'
        plot_path = os.path.join(model_dir, plot_filename)
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"矿点击中情况图已保存到: {plot_path}")
        
        # 关闭图形以释放内存
        plt.close()
        
        # 打印统计信息
        hit_count = sum(hit_status)
        total_count = len(hit_status)
        gdr = hit_count / total_count if total_count > 0 else 0
        
        print(f"\n击中情况统计:")
        print(f"  总矿点数: {total_count}")
        print(f"  被击中矿点数: {hit_count} (绿色圆点)")
        print(f"  未击中矿点数: {total_count - hit_count} (红色叉号)")
        print(f"  GDR: {gdr:.2%}")
        print(f"  异常网格数: {len(high_confidence_positions)}")
        
    except Exception as e:
        print(f"绘制矿点击中情况图时发生错误: {e}")
        import traceback
        traceback.print_exc()

def save_predictions(positions, predictions, confidences, labels, output_file):
    """保存预测结果为h5格式"""
    # 将预测结果转换为数值形式
    predictions_numeric = np.array([1 if p == 'Positive' else -1 for p in predictions], dtype=np.int8)
    
    # 将positions转换为NumPy数组
    try:
        # 如果positions是张量列表，先转换为NumPy数组
        if isinstance(positions[0], torch.Tensor):
            positions = [pos.cpu().numpy() for pos in positions]
        
        # 确保positions是二维数组
        positions_adjusted = np.vstack(positions).astype(np.float32)  # 使用float32类型
        positions_adjusted -= 8  # 减去偏移量
    except Exception as e:
        print(f"转换positions时出错: {e}")
        print(f"positions的第一个元素: {positions[0]}")
        print(f"positions的形状: {len(positions)}")
        raise
    
    # 保存为h5格式
    with h5py.File(output_file, 'w') as f:
        # 保存数据到根目录
        f.create_dataset('positions', data=positions_adjusted)
        f.create_dataset('predictions', data=predictions_numeric)
        f.create_dataset('confidences', data=np.array(confidences, dtype=np.float32))
        f.create_dataset('labels', data=np.array(labels, dtype=np.int8))
        
        # 添加元数据
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
    def __init__(self, data_file, label_file, model_dir, use_custom_norm=False):
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
        dataset = WindowDataset(args.input, args.label_file, model_dir, use_custom_norm=True)
        
        # 从数据中获取实际输入维度
        sample_data = dataset[0][0]  # 获取第一个样本
        input_channels = sample_data.shape[0]  # 通道数
        input_height = sample_data.shape[1]  # 高度
        input_width = sample_data.shape[2]  # 宽度
        actual_input_dim = input_channels * input_height * input_width
        
        print(f"检测到实际输入维度: {actual_input_dim} ({input_channels}×{input_height}×{input_width})")
        
        # 使用实际输入维度而不是固定尺寸
        input_dim = actual_input_dim
        model = load_model(
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
            args.confidence_threshold
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
        save_predictions(all_positions, all_predictions, all_confidences, all_labels_np, output_file) # 确保传递 all_labels_np
        
        # 绘制矿点击中情况图
        print("\n正在绘制矿点击中情况图...")
        plot_deposit_distribution(plot_info, model_dir, args.model_type, all_positions)
        
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

if __name__ == '__main__':
    main()
