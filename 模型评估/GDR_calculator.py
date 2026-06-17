#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDR计算器 - 地质多样性缩减指标计算工具

基于predict.py中的GDR计算逻辑，使用置信度阈值来计算金矿击中率
支持h5和dat文件格式
"""

import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import argparse
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(PROJECT_ROOT))

from model_comparison.metric_protocol import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_THRESHOLD_STEP,
    THRESHOLD_STRATEGY,
    deposit_hit_stats as protocol_deposit_hit_stats,
    independent_metric_rank,
    metric_protocol_fields,
    threshold_candidates,
)


def _read_coordinate_table(file_path: str) -> pd.DataFrame:
    """Read coordinate files used by the training/prediction workflow."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".txt", ".dat"}:
        try:
            frame = pd.read_csv(path, sep=None, engine="python")
            column_names = {str(col).strip().lower() for col in frame.columns}
            known_x = {"x", "coord_x", "point_x", "east", "easting", "longitude", "经度", "横坐标", "x坐标"}
            known_y = {"y", "coord_y", "point_y", "north", "northing", "latitude", "纬度", "纵坐标", "y坐标"}
            if column_names & known_x and column_names & known_y:
                return frame
        except Exception:
            pass
        return pd.read_csv(path, sep=r"[\s,]+", header=None, engine="python")

    raise ValueError(f"不支持的文件格式: {path.suffix}")


def _identify_xy_columns(frame: pd.DataFrame) -> Tuple[Optional[object], Optional[object]]:
    """Identify X/Y columns with the same tolerance as the new data flow."""
    column_map = {str(col).strip().lower(): col for col in frame.columns}
    x_keys = ("x", "coord_x", "point_x", "east", "easting", "longitude", "经度", "横坐标", "x坐标")
    y_keys = ("y", "coord_y", "point_y", "north", "northing", "latitude", "纬度", "纵坐标", "y坐标")
    x_col = next((column_map[key] for key in x_keys if key in column_map), None)
    y_col = next((column_map[key] for key in y_keys if key in column_map), None)

    if (x_col is None or y_col is None) and len(frame.columns) >= 2:
        x_col = frame.columns[0]
        y_col = frame.columns[1]
    return x_col, y_col

# ==================== 预设参数配置 ====================
# 在这里预设所有参数，可直接运行程序
PREDICTION_FILE = "label_plots/out.dat"  # 预测文件路径
DEPOSIT_FILE = "deposit.xlsx"                        # 矿点位置文件
DISTANCE_THRESHOLD = DEFAULT_DISTANCE_THRESHOLD                            # 距离阈值(米)
CONFIDENCE_THRESHOLD = 0.5                          # 置信度阈值


# H5文件的键名设置
POSITIONS_KEY = "positions"                         # H5文件中位置数据的键名
CONFIDENCES_KEY = "confidences"                     # H5文件中置信度数据的键名

# 运行模式设置
USE_COMMAND_LINE = False                           # 是否使用命令行参数（False=使用预设参数）
# =====================================================

class GDRCalculator:
    """GDR计算器 - 基于predict.py的计算逻辑"""
    
    def __init__(self, 
                 deposit_file: str = DEPOSIT_FILE,
                 distance_threshold: float = DISTANCE_THRESHOLD,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        初始化GDR计算器
        
        Args:
            deposit_file: 矿点位置文件路径
            distance_threshold: 击中判定距离阈值(米)
            confidence_threshold: 置信度阈值
        """
        self.deposit_file = deposit_file
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.deposits_df = None
        self.x_col = None
        self.y_col = None
        self.load_deposits()
        
    def load_deposits(self):
        """加载矿点位置数据"""
        try:
            if not Path(self.deposit_file).exists():
                raise FileNotFoundError(f"矿点文件不存在: {self.deposit_file}")
                
            # 支持多种文件格式
            self.deposits_df = _read_coordinate_table(self.deposit_file)
                
            print(f"成功读取矿点文件: {self.deposit_file}")
            print(f"矿点数据列名: {self.deposits_df.columns.tolist()}")
            print(f"矿点数据形状: {self.deposits_df.shape}")
            
            # 智能识别坐标列
            self.x_col, self.y_col = self._identify_coordinate_columns()
            if self.x_col is None or self.y_col is None:
                raise ValueError("无法识别坐标列，请检查文件格式")

            self.deposits_df[self.x_col] = pd.to_numeric(self.deposits_df[self.x_col], errors="coerce")
            self.deposits_df[self.y_col] = pd.to_numeric(self.deposits_df[self.y_col], errors="coerce")
            self.deposits_df = self.deposits_df.dropna(subset=[self.x_col, self.y_col]).reset_index(drop=True)
            if self.deposits_df.empty:
                raise ValueError("矿点文件中没有可用的 X/Y 坐标。")
                
            print(f"✓ 成功加载 {len(self.deposits_df)} 个矿点位置")
            print(f"✓ 坐标列识别: X={self.x_col}, Y={self.y_col}")
            print(f"✓ 距离阈值设置: {self.distance_threshold} 米")
            print(f"✓ 置信度阈值设置: {self.confidence_threshold}")
            
        except Exception as e:
            print(f"✗ 加载矿点文件失败: {e}")
            raise
            
    def _identify_coordinate_columns(self) -> Tuple[Optional[str], Optional[str]]:
        """智能识别坐标列名 - 基于predict.py的逻辑"""
        return _identify_xy_columns(self.deposits_df)
        
    def load_predictions_from_dat(self, dat_path: str) -> Dict:
        """从DAT文件加载预测结果"""
        try:
            # 读取dat文件，不使用表头
            df = pd.read_csv(dat_path, sep=r'\s+', header=None, encoding='utf-8')
            
            print(f"检测到DAT文件列数: {len(df.columns)}")
            
            # 根据列数判断文件格式
            if len(df.columns) == 3:
                # 3列格式: X坐标, Y坐标, 置信度/概率值
                df.columns = ['x_coord', 'y_coord', 'confidence']
                
                # 提取数据
                x_coords = df['x_coord'].values.astype(np.float32)
                y_coords = df['y_coord'].values.astype(np.float32)
                confidences = df['confidence'].values.astype(np.float32)
                
                # 将置信度归一化到[0,1]范围
                if np.max(confidences) > 1.0:
                    print(f"置信度范围: [{np.min(confidences):.2e}, {np.max(confidences):.2e}]")
                    print("检测到置信度值很大，进行归一化处理...")
                    # 使用sigmoid函数将大数值映射到[0,1]范围
                    # 对于很大的数值，sigmoid会接近1
                    confidences = 1.0 / (1.0 + np.exp(-np.clip(confidences, -500, 500)))
                    print(f"归一化后置信度范围: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")
                
                # 生成预测结果（基于置信度阈值）
                predictions = (confidences > 0.5).astype(np.int32)
                predictions[predictions == 0] = -1  # 转换为-1/1格式
                
            elif len(df.columns) == 4:
                # 4列格式: X坐标, Y坐标, 预测值, 置信度
                df.columns = ['x_coord', 'y_coord', 'prediction', 'confidence']
                
                # 提取数据
                x_coords = df['x_coord'].values.astype(np.float32)
                y_coords = df['y_coord'].values.astype(np.float32)
                predictions = df['prediction'].values.astype(np.int32)
                confidences = df['confidence'].values.astype(np.float32)
                
            else:
                raise ValueError(f"不支持的DAT文件格式，列数: {len(df.columns)}，支持3列或4列格式")
            
            # 创建positions列表（与predict.py保持一致）
            positions = []
            for i in range(len(x_coords)):
                positions.append(np.array([x_coords[i], y_coords[i]]))
            
            print(f"✓ 成功从DAT文件加载 {len(confidences)} 个预测点")
            print(f"✓ 文件格式: {len(df.columns)}列")
            print(f"✓ 置信度范围: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")
            
            return {
                'positions': positions,
                'confidences': confidences,
                'predictions': predictions,
                'format': 'dat',
                'path': dat_path
            }
            
        except Exception as e:
            print(f"✗ 加载DAT文件失败: {e}")
            raise
            
    def load_predictions_from_h5(self, h5_path: str, 
                                positions_key: str = POSITIONS_KEY,
                                confidences_key: str = CONFIDENCES_KEY) -> Dict:
        """从H5文件加载预测结果"""
        try:
            with h5py.File(h5_path, 'r') as f:
                # 读取位置数据
                resolved_positions_key = positions_key if positions_key in f else None
                if resolved_positions_key is None:
                    resolved_positions_key = next((key for key in ("positions", "coordinates", "coords") if key in f), None)
                if resolved_positions_key is None:
                    raise KeyError("H5预测文件缺少 positions/coordinates 坐标数据集。")
                positions_data = f[resolved_positions_key][:]
                
                # 读取置信度数据
                resolved_confidences_key = confidences_key if confidences_key in f else None
                if resolved_confidences_key is None:
                    resolved_confidences_key = next((key for key in ("confidences", "confidence", "probabilities", "scores") if key in f), None)
                if resolved_confidences_key is None:
                    raise KeyError("H5预测文件缺少 confidences/probabilities/scores 置信度数据集。")
                confidences = f[resolved_confidences_key][:].astype(np.float32)
                
                # 创建positions列表（与predict.py保持一致）
                positions = []
                for i in range(len(positions_data)):
                    positions.append(positions_data[i])
                
                # 如果有预测结果，也读取
                predictions = None
                if 'predictions' in f:
                    predictions = f['predictions'][:].astype(np.int32)

                test_mask = None
                if 'test_mask' in f:
                    test_mask = f['test_mask'][:].astype(bool)

                test_indices = None
                if 'test_indices' in f:
                    test_indices = f['test_indices'][:].astype(np.int64)
                metadata = dict(f.attrs)
                
            print(f"✓ 成功从H5文件加载 {len(confidences)} 个预测点")
            
            return {
                'positions': positions,
                'confidences': confidences,
                'predictions': predictions,
                'test_mask': test_mask,
                'test_indices': test_indices,
                'metadata': metadata,
                'format': 'h5',
                'path': h5_path
            }
            
        except Exception as e:
            print(f"✗ 加载H5文件失败: {e}")
            raise
            
    def load_predictions(self, file_path: str, **kwargs) -> Dict:
        """自动识别文件格式并加载预测结果"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"预测文件不存在: {file_path}")
            
        if file_path.suffix.lower() == '.dat':
            return self.load_predictions_from_dat(str(file_path))
        elif file_path.suffix.lower() in ['.h5', '.hdf5']:
            return self.load_predictions_from_h5(str(file_path), **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
    def calculate_gdr(self, prediction_data: Dict) -> Tuple[float, int, int, Dict]:
        """
        计算GDR (Gold Deposit Hit Rate) 指标 - 基于predict.py的逻辑
        
        Args:
            prediction_data: 预测数据字典，包含positions和confidences
            
        Returns:
            gdr: GDR指标值 (0-1)
            hit_deposits: 被击中的矿点数量
            total_deposits: 总矿点数量
            plot_info: 绘图信息
        """
        positions = prediction_data['positions']
        confidences = prediction_data['confidences']
        
        # 获取矿点坐标
        deposit_coords = self.deposits_df[[self.x_col, self.y_col]].values
        total_deposits = len(deposit_coords)
        print(f"总矿点数量: {total_deposits}")
        print(f"前5个矿点坐标: {deposit_coords[:5]}")
        
        # 获取置信度大于阈值的网格位置
        high_confidence_positions = []
        for i, conf in enumerate(confidences):
            if conf > self.confidence_threshold:
                if isinstance(positions[i], torch.Tensor):
                    pos = positions[i].cpu().numpy()
                else:
                    pos = positions[i]
                # 新预测模块保存的 positions 已经是用于评估的坐标，不再做固定偏移。
                high_confidence_positions.append(np.asarray(pos, dtype=np.float64)[:2])
        
        print(f"置信度>{self.confidence_threshold}的网格数量: {len(high_confidence_positions)}")
        
        if len(high_confidence_positions) == 0:
            return 0.0, 0, total_deposits, None
        
        high_confidence_positions = np.asarray(high_confidence_positions, dtype=np.float32)

        # 计算每个矿点到最近高置信度网格的距离
        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:  # SciPy may be unavailable in certain deployments
            cKDTree = None

        if cKDTree is not None and len(high_confidence_positions) >= 10:
            tree = cKDTree(high_confidence_positions)
            min_distances, _ = tree.query(deposit_coords, k=1)
        else:
            min_distances = np.empty(total_deposits, dtype=np.float32)
            for i, deposit_coord in enumerate(deposit_coords):
                distances = np.sqrt(np.sum((high_confidence_positions - deposit_coord) ** 2, axis=1))
                min_distances[i] = distances.min() if len(distances) > 0 else float('inf')

        # 计算击中情况
        hit_mask = min_distances <= self.distance_threshold
        hit_deposits = int(np.sum(hit_mask))
        hit_status = hit_mask.tolist()
        hit_details = []
        for i, (status, min_distance, deposit_coord) in enumerate(zip(hit_status, min_distances, deposit_coords)):
            deposit_x, deposit_y = deposit_coord
            state_text = "击中" if status else "未击中"
            hit_details.append(
                f"矿点{i+1} ({deposit_x:.1f}, {deposit_y:.1f}): {state_text} (最近距离: {min_distance:.1f}m)"
            )
        
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
            'threshold_distance': self.distance_threshold,
            'confidence_threshold': self.confidence_threshold
        }
        
        return gdr, hit_deposits, total_deposits, plot_info

    def _positions_array(self, prediction_data: Dict) -> np.ndarray:
        positions = prediction_data["positions"]
        if len(positions) == 0:
            return np.empty((0, 2), dtype=np.float64)
        rows = []
        for pos in positions:
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
            rows.append(np.asarray(pos, dtype=np.float64)[:2])
        return np.asarray(rows, dtype=np.float64)

    def _coordinate_mask(self, prediction_positions: np.ndarray, region_positions: np.ndarray) -> np.ndarray:
        if len(region_positions) == 0:
            return np.zeros(len(prediction_positions), dtype=bool)
        rounded_region = {
            (round(float(x), 6), round(float(y), 6))
            for x, y in np.asarray(region_positions, dtype=np.float64)[:, :2]
        }
        return np.asarray(
            [
                (round(float(x), 6), round(float(y), 6)) in rounded_region
                for x, y in prediction_positions[:, :2]
            ],
            dtype=bool,
        )

    def load_test_area_mask(self, prediction_data: Dict, area_file: Optional[str] = None) -> np.ndarray:
        """Load a boolean mask defining which predictions belong to the independent test area."""
        positions = self._positions_array(prediction_data)
        total_count = len(positions)
        if total_count == 0:
            raise ValueError("预测文件中没有可用坐标。")

        if area_file:
            area_path = Path(area_file)
            if not area_path.exists():
                raise FileNotFoundError(f"测试区域文件不存在: {area_file}")
            suffix = area_path.suffix.lower()
            if suffix in {".h5", ".hdf5"}:
                with h5py.File(area_path, "r") as f:
                    key = next((name for name in ("positions", "coordinates", "coords") if name in f), None)
                    if key is not None:
                        return self._coordinate_mask(positions, np.asarray(f[key][:], dtype=np.float64))
                    key = next((name for name in ("test_mask", "mask", "area_mask") if name in f), None)
                    if key is not None:
                        mask = np.asarray(f[key][:]).astype(bool).reshape(-1)
                        if len(mask) != total_count:
                            raise ValueError(f"测试区域mask长度不匹配: mask={len(mask)}, predictions={total_count}")
                        return mask
                    key = next((name for name in ("test_indices", "indices", "area_indices") if name in f), None)
                    if key is not None:
                        indices = np.asarray(f[key][:]).astype(np.int64).reshape(-1)
                        mask = np.zeros(total_count, dtype=bool)
                        mask[indices[(indices >= 0) & (indices < total_count)]] = True
                        return mask
                    key = next((name for name in ("positions", "coordinates", "coords") if name in f), None)
                    if key is not None:
                        return self._coordinate_mask(positions, np.asarray(f[key][:], dtype=np.float64))
                    raise KeyError("测试区域H5需要包含 test_mask/test_indices/positions 等数据集。")
            if suffix in {".npy", ".npz"}:
                data = np.load(area_path, allow_pickle=False)
                if isinstance(data, np.lib.npyio.NpzFile):
                    key = next((name for name in ("test_mask", "mask", "test_indices", "indices", "positions", "coordinates") if name in data), None)
                    if key is None:
                        raise KeyError("测试区域NPZ需要包含 test_mask/test_indices/positions 等数组。")
                    array = data[key]
                else:
                    array = data
                array = np.asarray(array)
                if array.ndim == 1 and len(array) == total_count and array.dtype != np.float64:
                    if np.isin(array, [0, 1, False, True]).all():
                        return array.astype(bool)
                if array.ndim == 1:
                    indices = array.astype(np.int64)
                    mask = np.zeros(total_count, dtype=bool)
                    mask[indices[(indices >= 0) & (indices < total_count)]] = True
                    return mask
                return self._coordinate_mask(positions, array)

            frame = _read_coordinate_table(str(area_path))
            x_col, y_col = _identify_xy_columns(frame)
            if x_col is None or y_col is None:
                raise ValueError("测试区域坐标文件无法识别X/Y列。")
            coords = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=np.float64)
            return self._coordinate_mask(positions, coords)

        if prediction_data.get("test_mask") is not None:
            mask = np.asarray(prediction_data["test_mask"]).astype(bool).reshape(-1)
            if len(mask) == total_count:
                return mask
        if prediction_data.get("test_indices") is not None:
            indices = np.asarray(prediction_data["test_indices"]).astype(np.int64).reshape(-1)
            mask = np.zeros(total_count, dtype=bool)
            mask[indices[(indices >= 0) & (indices < total_count)]] = True
            return mask

        raise ValueError("独立测试集评价需要测试区域文件，或预测H5内置 test_mask/test_indices。")

    def _deposit_hit_stats(self, selected_positions: np.ndarray, deposit_coords: np.ndarray) -> Tuple[int, np.ndarray, list]:
        return protocol_deposit_hit_stats(
            selected_positions,
            deposit_coords,
            distance_threshold=self.distance_threshold,
        )

    def calculate_independent_test_metrics(
        self,
        prediction_data: Dict,
        test_area_mask: np.ndarray,
        threshold_step: float = 0.01,
        fixed_threshold: Optional[float] = None,
    ) -> Dict:
        """Evaluate only independent test deposits and the independent test area."""
        positions = self._positions_array(prediction_data)
        confidences = np.asarray(prediction_data["confidences"], dtype=np.float64).reshape(-1)
        if len(positions) != len(confidences):
            raise ValueError(f"预测坐标与置信度数量不一致: positions={len(positions)}, confidences={len(confidences)}")

        mask = np.asarray(test_area_mask).astype(bool).reshape(-1)
        if len(mask) != len(confidences):
            raise ValueError(f"测试区域mask长度不匹配: mask={len(mask)}, predictions={len(confidences)}")
        test_positions = positions[mask]
        test_confidences = confidences[mask]
        if len(test_positions) == 0:
            raise ValueError("测试区域内没有预测点，无法计算独立测试集指标。")

        deposit_coords = self.deposits_df[[self.x_col, self.y_col]].to_numpy(dtype=np.float64)
        total_deposits = len(deposit_coords)
        if total_deposits == 0:
            raise ValueError("独立测试矿点文件中没有可用矿点。")

        if fixed_threshold is not None:
            thresholds = np.asarray([float(fixed_threshold)], dtype=np.float64)
        else:
            thresholds = threshold_candidates(step=float(threshold_step or DEFAULT_THRESHOLD_STEP))

        best = None
        for threshold in thresholds:
            selected = test_positions[test_confidences > threshold]
            pa = len(selected) / len(test_positions)
            hit_count, min_distances, hit_status = self._deposit_hit_stats(selected, deposit_coords)
            sr = hit_count / total_deposits if total_deposits else 0.0
            ei = sr / pa if pa > 0 else 0.0
            current = {
                "threshold": float(threshold),
                "sr": float(sr),
                "pa": float(pa),
                "paf": float(pa),
                "ei": float(ei),
                "test_sr": float(sr),
                "test_paf": float(pa),
                "test_ei": float(ei),
                "hit_deposits": int(hit_count),
                "total_deposits": int(total_deposits),
                "test_detected_count": int(hit_count),
                "test_mineral_count": int(total_deposits),
                "high_potential_count": int(len(selected)),
                "test_area_count": int(len(test_positions)),
                "min_distances": min_distances,
                "hit_status": hit_status,
            }
            if best is None or independent_metric_rank(current) > independent_metric_rank(best):
                best = current

        if best is None:
            raise RuntimeError("阈值扫描失败，未生成任何测试集指标。")
        best["mean_min_distance"] = float(np.mean(best["min_distances"])) if len(best["min_distances"]) else 0.0
        best["median_min_distance"] = float(np.median(best["min_distances"])) if len(best["min_distances"]) else 0.0
        best["test_area_mask"] = mask
        best.update(
            metric_protocol_fields(
                threshold_step=float(threshold_step or DEFAULT_THRESHOLD_STEP),
                distance_threshold=float(self.distance_threshold),
                threshold_strategy="fixed" if fixed_threshold is not None else THRESHOLD_STRATEGY,
            )
        )
        return best
    


def print_header():
    """打印程序标题"""
    print("\n" + "="*60)
    print("🎯 GDR计算器 - 基于predict.py的计算逻辑")
    print("="*60)
    if USE_COMMAND_LINE:
        print("📊 运行模式: 命令行参数模式")
    else:
        print("📊 运行模式: 预设参数模式")
        print(f"📁 预设文件: {PREDICTION_FILE}")
        print(f"🏠 矿点文件: {DEPOSIT_FILE}")
        print(f"📏 距离阈值: {DISTANCE_THRESHOLD}米")
        print(f"🎯 置信度阈值: {CONFIDENCE_THRESHOLD}")
        print("📈 输出方式: 仅屏幕显示")
    print("="*60)

def run_with_preset_params():
    """使用预设参数运行GDR计算"""
    print_header()
    
    try:
        # 初始化计算器
        calculator = GDRCalculator(
            deposit_file=DEPOSIT_FILE,
            distance_threshold=DISTANCE_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        # 加载预测数据
        print(f"\n📂 加载预测文件: {PREDICTION_FILE}")
        prediction_data = calculator.load_predictions(
            PREDICTION_FILE,
            positions_key=POSITIONS_KEY,
            confidences_key=CONFIDENCES_KEY
        )
        
        # 生成模型名称
        model_name = Path(PREDICTION_FILE).stem
        
        # 计算GDR
        print(f"\n🎯 计算GDR指标...")
        gdr, hit_deposits, total_deposits, plot_info = calculator.calculate_gdr(prediction_data)
        
        # 显示结果
        print(f"\n📈 GDR计算结果:")
        print(f"  GDR (金矿击中率): {gdr:.4f} ({gdr*100:.2f}%)")
        print(f"  被击中矿点数: {hit_deposits}")
        print(f"  总矿点数: {total_deposits}")
        print(f"  击中率: {hit_deposits}/{total_deposits}")
        print(f"  距离阈值: {DISTANCE_THRESHOLD}m")
        print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
        
        print(f"\n✅ GDR计算完成!")
        return 0
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """主函数 - 根据设置选择运行模式"""
    
    if not USE_COMMAND_LINE:
        # 使用预设参数模式
        return run_with_preset_params()
    
    # 使用命令行参数模式
    parser = argparse.ArgumentParser(description='GDR计算器 - 基于predict.py逻辑')
    parser.add_argument('prediction_file', help='预测结果文件路径 (.dat或.h5)')
    parser.add_argument('--deposit_file', default=DEPOSIT_FILE, help='矿点位置文件')
    parser.add_argument('--distance_threshold', type=float, default=DISTANCE_THRESHOLD, 
                       help='距离阈值(米)')
    parser.add_argument('--confidence_threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help='置信度阈值')
    parser.add_argument('--positions_key', default=POSITIONS_KEY, help='H5文件中位置数据的键名')
    parser.add_argument('--confidences_key', default=CONFIDENCES_KEY, help='H5文件中置信度数据的键名')
    
    args = parser.parse_args()
    
    print_header()
    
    try:
        # 初始化计算器
        calculator = GDRCalculator(
            deposit_file=args.deposit_file,
            distance_threshold=args.distance_threshold,
            confidence_threshold=args.confidence_threshold
        )
        
        # 加载预测数据
        print(f"\n📂 加载预测文件: {args.prediction_file}")
        prediction_data = calculator.load_predictions(
            args.prediction_file,
            positions_key=args.positions_key,
            confidences_key=args.confidences_key
        )
        
        # 生成模型名称
        model_name = Path(args.prediction_file).stem
        
        # 计算GDR
        print(f"\n🎯 计算GDR指标...")
        gdr, hit_deposits, total_deposits, plot_info = calculator.calculate_gdr(prediction_data)
        
        # 显示结果
        print(f"\n📈 GDR计算结果:")
        print(f"  GDR (金矿击中率): {gdr:.4f} ({gdr*100:.2f}%)")
        print(f"  被击中矿点数: {hit_deposits}")
        print(f"  总矿点数: {total_deposits}")
        print(f"  击中率: {hit_deposits}/{total_deposits}")
        print(f"  距离阈值: {args.distance_threshold}m")
        print(f"  置信度阈值: {args.confidence_threshold}")
        
    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        return 1
        
    print(f"\n✅ GDR计算完成!")
    return 0

if __name__ == "__main__":
    exit(main()) 
