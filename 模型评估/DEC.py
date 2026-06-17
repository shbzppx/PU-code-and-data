#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEC dynamic efficiency curve generator.

Load DAT files and generate model comparison DEC curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
import warnings
from model_name_utils import get_model_display_name, normalize_model_key
warnings.filterwarnings('ignore')

# 璁剧疆涓枃瀛椾綋
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 棰勮鍙傛暟閰嶇疆 ====================
#DISTANCE_THRESHOLD = np.sqrt(2.0)    # 璺濈闃堝€?(闂磋窛)
DISTANCE_THRESHOLD = 4.0    # 璺濈闃堝€?(闂磋窛)
PAR_MIN = 0.0                 # PAR鏈€灏忓€?(%)
PAR_MAX = 100.0               # PAR鏈€澶у€?(%)
STEP = 1.0                    # PAR姝ラ暱 (%)
OUTPUT_DIR = "DEC"            # 杈撳嚭鐩綍
DEFAULT_DEPOSIT_FILE = "deposit.xlsx" # 鐭跨偣鏂囦欢
# =====================================================
class SimpleDECGenerator:
    """Simple DEC curve generator."""
    
    def __init__(self, deposit_file: Optional[str] = None):
        """Initialize the generator."""
        self.deposits_df = None
        self.x_col = None
        self.y_col = None
        self.model_base_pars = {}  # 瀛樺偍姣忎釜妯″瀷鐨勫熀鍑哖AR
        self.deposit_file = deposit_file or DEFAULT_DEPOSIT_FILE
        self.load_deposits()
        
    def load_deposits(self):
        """鍔犺浇鐭跨偣浣嶇疆鏁版嵁"""
        try:
            deposit_path = Path(self.deposit_file)
            if not deposit_path.exists():
                raise FileNotFoundError(f"鐭跨偣鏂囦欢涓嶅瓨鍦? {deposit_path}")

            suffix = deposit_path.suffix.lower()
            if suffix in {".xlsx", ".xls"}:
                self.deposits_df = pd.read_excel(deposit_path)
            elif suffix == ".csv":
                self.deposits_df = pd.read_csv(deposit_path)
            elif suffix == ".tsv":
                self.deposits_df = pd.read_csv(deposit_path, sep="\t")
            elif suffix in {".txt", ".dat"}:
                try:
                    self.deposits_df = pd.read_csv(deposit_path, sep=None, engine="python")
                except Exception:
                    self.deposits_df = pd.read_csv(deposit_path, sep=r"[\s,]+", header=None, engine="python")
            else:
                raise ValueError(f"不支持的矿点文件格式: {suffix}")
            
            # 智能识别坐标列
            self.x_col, self.y_col = self._identify_coordinate_columns()
            if self.x_col is None or self.y_col is None:
                raise ValueError("无法识别坐标列，请检查矿点文件格式")
            self.deposits_df[self.x_col] = pd.to_numeric(self.deposits_df[self.x_col], errors="coerce")
            self.deposits_df[self.y_col] = pd.to_numeric(self.deposits_df[self.y_col], errors="coerce")
            self.deposits_df = self.deposits_df.dropna(subset=[self.x_col, self.y_col]).reset_index(drop=True)
            if self.deposits_df.empty:
                raise ValueError("矿点文件中没有可用的 X/Y 坐标")
                
            print(f"成功加载 {len(self.deposits_df)} 个矿点位置")
            print(f"鉁?鍧愭爣鍒楄瘑鍒? X={self.x_col}, Y={self.y_col}")
            
        except Exception as e:
            print(f"鉁?鍔犺浇鐭跨偣鏂囦欢澶辫触: {e}")
            raise
            
    def _identify_coordinate_columns(self) -> Tuple[Optional[str], Optional[str]]:
        """鏅鸿兘璇嗗埆鍧愭爣鍒楀悕"""
        columns = self.deposits_df.columns.tolist()
        
        # 鍙兘鐨刋鍧愭爣鍒楀悕
        x_patterns = ['x', 'X', '经度', 'longitude', 'lon', 'lng', 'east', 'eastings']
        # 鍙兘鐨刌鍧愭爣鍒楀悕  
        y_patterns = ['y', 'Y', '纬度', 'latitude', 'lat', 'north', 'northings']
        
        x_col = None
        y_col = None
        
        # 查找 X 坐标列
        for col in columns:
            if any(pattern in str(col).lower() for pattern in [p.lower() for p in x_patterns]):
                x_col = col
                break
                
        # 查找 Y 坐标列
        for col in columns:
            if any(pattern in str(col).lower() for pattern in [p.lower() for p in y_patterns]):
                y_col = col
                break

        if (x_col is None or y_col is None) and len(columns) >= 2:
            x_col = columns[0]
            y_col = columns[1]

        return x_col, y_col
        
    def load_model_predictions(self, model_path: str, model_name: str) -> Dict:
        """Load DAT-format model predictions."""
        try:
            # 读取 DAT 文件，跳过表头
            df = pd.read_csv(model_path, sep=r'\s+', header=0, encoding='utf-8')
            
            # 閲嶅懡鍚嶅垪浠ヤ究澶勭悊
            df.columns = ['x_coord', 'y_coord', 'prediction', 'confidence']
            
            # 鎻愬彇鏁版嵁
            x_coords = df['x_coord'].values.astype(np.float32)
            y_coords = df['y_coord'].values.astype(np.float32)
            probabilities = df['confidence'].values.astype(np.float32)
            
            print(f"成功加载模型 '{model_name}': {len(probabilities)} 个预测点")
            
            return {
                'name': model_name,
                'x_coords': x_coords,
                'y_coords': y_coords,
                'probabilities': probabilities,
                'path': model_path
            }
            
        except Exception as e:
            print(f"加载模型失败 '{model_name}': {e}")
            raise
            
    def calculate_base_par(self, model_data: Dict) -> float:
        """璁＄畻妯″瀷鍦ㄧ疆淇″害0.5涓嬬殑闃堝€糚AR锛圓鍊硷級鍙婂搴旂殑GDR"""
        probabilities = model_data['probabilities']
        
        # 计算置信度大于 0.5 的网格占比
        high_confidence_mask = probabilities > 0.5
        high_confidence_count = np.sum(high_confidence_mask)
        total_grids = len(probabilities)
        
        # 闃堝€糚AR锛圓鍊硷級= 楂樼疆淇″害缃戞牸鍗犳€荤綉鏍肩殑姣斾緥
        threshold_par = (high_confidence_count / total_grids) * 100.0
        
        # 璁＄畻闃堝€糚AR瀵瑰簲鐨凣DR
        if high_confidence_count > 0:
            # 获取高置信度网格坐标
            x_coords = model_data['x_coords']
            y_coords = model_data['y_coords']
            high_confidence_indices = np.where(high_confidence_mask)[0]
            
            # 使用固定随机种子保证结果可复现
            np.random.seed(42)
            selected_indices = high_confidence_indices
            
            # 杞崲涓簄umpy鏁扮粍
            selected_indices = np.array(selected_indices)
            
            # 鑾峰彇閫変腑缃戞牸鐨勫潗鏍囷紙娣诲姞涓巔redict.py鐩稿悓鐨勫亸绉昏皟鏁达級
            selected_x = x_coords[selected_indices] - 8  # 娣诲姞鍋忕Щ璋冩暣
            selected_y = y_coords[selected_indices] - 8  # 娣诲姞鍋忕Щ璋冩暣
            
            # 璁＄畻鏈夊灏戠熆鐐硅鍑讳腑
            hit_count = 0
            total_deposits = len(self.deposits_df)
            
            for _, deposit in self.deposits_df.iterrows():
                deposit_x = deposit[self.x_col]
                deposit_y = deposit[self.y_col]
                
                # 计算到所有选中网格的距离
                distances = np.sqrt((selected_x - deposit_x)**2 + (selected_y - deposit_y)**2)
                
                # 如果最近距离小于阈值，则视为命中
                if np.min(distances) <= DISTANCE_THRESHOLD:
                    hit_count += 1
                    
            # 璁＄畻GDR
            threshold_gdr = (hit_count / total_deposits) * 100.0 if total_deposits > 0 else 0.0
        else:
            threshold_gdr = 0.0
        
        print(f"妯″瀷 '{model_data['name']}' 闃堝€糚AR(A): {threshold_par:.2f}% -> GDR: {threshold_gdr:.2f}% (楂樼疆淇″害缃戞牸: {high_confidence_count}/{total_grids})")
        
        return threshold_par
    
    def calculate_gdr_at_par(self, model_data: Dict, par_percent: float) -> float:
        """璁＄畻鎸囧畾PAR涓嬬殑GDR - 鍩轰簬闃堝€糚AR鐨勪笁闃舵閫夋嫨绛栫暐"""
        # 鑾峰彇鏁版嵁
        x_coords = model_data['x_coords']
        y_coords = model_data['y_coords']
        probabilities = model_data['probabilities']
        
        # 特殊处理 PAR=0 的情况
        if par_percent == 0.0:
            return 0.0
        
        # 鑾峰彇鎴栬绠楅槇鍊糚AR锛圓鍊硷級
        model_name = model_data['name']
        if model_name not in self.model_base_pars:
            self.model_base_pars[model_name] = self.calculate_base_par(model_data)
        
        threshold_par_A = self.model_base_pars[model_name]
        
        # 璁＄畻闇€瑕侀€夋嫨鐨勭綉鏍兼€绘暟
        total_grids = len(probabilities)
        top_k = int(total_grids * par_percent / 100.0)
        
        if top_k == 0:
            return 0.0
        
        # 鎵惧嚭缃俊搴﹀ぇ浜?.5鐨勭綉鏍煎拰鍏朵粬缃戞牸
        high_confidence_mask = probabilities > 0.5
        high_confidence_indices = np.where(high_confidence_mask)[0]
        low_confidence_indices = np.where(~high_confidence_mask)[0]
        
        selected_indices = []
        
        # 浣跨敤娴偣鏁版瘮杈冿紝閬垮厤绮惧害闂
        if abs(par_percent - threshold_par_A) < 0.001:  # PAR = 闃堝€糚AR(A)
            # 情况2：选择所有置信度>0.5的网格
            selected_indices = high_confidence_indices.tolist()
            print(f"  PAR={par_percent:.1f}% = 阈值 PAR(A)={threshold_par_A:.2f}%，选择全部高置信度网格 {len(selected_indices)} 个")
        
        elif par_percent < threshold_par_A:  # PAR < 闃堝€糚AR(A)
            # 情况1：从高置信度网格中选择占总体 PAR 比例的数量
            if len(high_confidence_indices) > 0:
                # 闇€瑕侀€夋嫨鐨勬暟閲?= 鎬荤綉鏍兼暟 脳 PAR%
                needed_count = top_k
                actual_count = min(needed_count, len(high_confidence_indices))
                
                # 从高置信度网格中随机选择
                np.random.seed(42)
                selected_indices = np.random.choice(high_confidence_indices, size=actual_count, replace=False).tolist()
                print(f"  PAR={par_percent:.1f}% < 阈值 PAR(A)={threshold_par_A:.2f}%，从 {len(high_confidence_indices)} 个高置信度网格中随机选择 {actual_count} 个")
            else:
                # 濡傛灉娌℃湁楂樼疆淇″害缃戞牸锛屾寜姒傜巼鎺掑簭閫夊彇
                sorted_indices = np.argsort(probabilities)[::-1]
                selected_indices = sorted_indices[:top_k].tolist()
                print(f"  PAR={par_percent:.1f}%，无高置信度网格，按概率排序选择前 {top_k} 个")
                
        else:  # PAR > 闃堝€糚AR(A)
            # 鎯呭喌3锛氫袱闃舵閫夋嫨绛栫暐
            if len(high_confidence_indices) > 0:
                # 第一阶段：选择全部高置信度网格
                selected_indices.extend(high_confidence_indices.tolist())
                remaining_needed = top_k - len(high_confidence_indices)
                
                print(f"  PAR={par_percent:.1f}% > 阈值 PAR(A)={threshold_par_A:.2f}%，第一阶段选择全部 {len(high_confidence_indices)} 个高置信度网格")
                
                # 绗簩闃舵锛氫粠鍓╀綑缃戞牸涓寜缃俊搴︿粠澶у埌灏忛€夊彇
                if remaining_needed > 0 and len(low_confidence_indices) > 0:
                    # 瀵逛綆缃俊搴︾綉鏍兼寜姒傜巼浠庨珮鍒颁綆鎺掑簭
                    low_confidence_probs = probabilities[low_confidence_indices]
                    sorted_low_indices = low_confidence_indices[np.argsort(low_confidence_probs)[::-1]]
                    
                    # 选择 remaining_needed 个样本
                    additional_count = min(remaining_needed, len(sorted_low_indices))
                    additional_indices = sorted_low_indices[:additional_count]
                    selected_indices.extend(additional_indices.tolist())
                    
                    print(f"  第二阶段从 {len(low_confidence_indices)} 个低置信度网格中按概率排序选择 {additional_count} 个")
                elif remaining_needed > 0:
                    print(f"  还需要额外选择 {remaining_needed} 个网格，但没有可用的低置信度网格")
            else:
                # 濡傛灉娌℃湁楂樼疆淇″害缃戞牸锛岀洿鎺ユ寜姒傜巼鎺掑簭閫夊彇
                sorted_indices = np.argsort(probabilities)[::-1]
                selected_indices = sorted_indices[:top_k].tolist()
                print(f"  PAR={par_percent:.1f}%，无高置信度网格，按概率排序选择前 {top_k} 个")
        
        # 杞崲涓簄umpy鏁扮粍
        selected_indices = np.array(selected_indices)
        
        # 获取选中网格的坐标
        selected_x = x_coords[selected_indices]
        selected_y = y_coords[selected_indices]
        
        # 娣诲姞鍧愭爣鍋忕Щ璋冩暣锛堜笌predict.py淇濇寔涓€鑷达級
        selected_x = selected_x - 8
        selected_y = selected_y - 8
        
        # 娣诲姞璋冭瘯杈撳嚭
        print(f"  璋冭瘯淇℃伅: 閫変腑缃戞牸鏁?{len(selected_indices)}, 鍧愭爣鑼冨洿: X[{selected_x.min():.1f}, {selected_x.max():.1f}], Y[{selected_y.min():.1f}, {selected_y.max():.1f}]")
        
        # 璁＄畻鏈夊灏戠熆鐐硅鍑讳腑
        hit_count = 0
        total_deposits = len(self.deposits_df)
        
        for _, deposit in self.deposits_df.iterrows():
            deposit_x = deposit[self.x_col]
            deposit_y = deposit[self.y_col]
            
            # 计算到所有选中网格的距离
            distances = np.sqrt((selected_x - deposit_x)**2 + (selected_y - deposit_y)**2)
            
            # 如果最近距离小于阈值，则视为命中
            min_distance = np.min(distances)
            if min_distance <= DISTANCE_THRESHOLD:
                hit_count += 1
                
        # 娣诲姞鏇村璋冭瘯淇℃伅
        print(f"  璋冭瘯淇℃伅: 鍑讳腑鐭跨偣鏁?{hit_count}/{total_deposits}, 璺濈闃堝€?{DISTANCE_THRESHOLD}")
        
        # 璁＄畻GDR
        gdr = (hit_count / total_deposits) * 100.0 if total_deposits > 0 else 0.0
        return gdr
        
    def generate_dec_curve(self, model_data: Dict) -> Tuple[List[float], List[float]]:
        """Generate DEC curve points for a single model."""
        print(f"姝ｅ湪鐢熸垚妯″瀷 '{model_data['name']}' 鐨凞EC鏇茬嚎...")
        
        par_list = []
        gdr_list = []
        
        # 鐢熸垚PAR搴忓垪
        current_par = PAR_MIN
        while current_par <= PAR_MAX:
            gdr = self.calculate_gdr_at_par(model_data, current_par)
            par_list.append(current_par)
            gdr_list.append(gdr)
            
            print(f"  PAR = {current_par:.1f}%, GDR = {gdr:.1f}%")
            current_par += STEP
            
        return par_list, gdr_list
        
    def plot_dec_curves(self, models_data: List[Dict], output_path: str):
        """缁樺埗澶氫釜妯″瀷鐨凞EC鏇茬嚎瀵规瘮鍥?- ROC椋庢牸"""
        plt.figure(figsize=(10, 10))  # 姝ｆ柟褰㈢敾甯?        
        # 预定义颜色列表
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # 存储所有曲线数据用于保存
        all_curves_data = {}
        
        for i, model_data in enumerate(models_data):
            # 鐢熸垚DEC鏇茬嚎
            par_list, gdr_list = self.generate_dec_curve(model_data)
            
            # 杞崲涓?-1鑼冨洿
            par_norm = [p / 100.0 for p in par_list]
            gdr_norm = [g / 100.0 for g in gdr_list]
            
            # 閫夋嫨棰滆壊
            color = colors[i % len(colors)]
            
            # 绘制曲线
            plt.plot(par_norm, gdr_norm, marker='o', markersize=3, linewidth=3, 
                    color=color, label=model_data['name'], alpha=0.8)
                    
            # 保存曲线数据
            all_curves_data[model_data['name']] = {
                'PAR': par_list,
                'GDR': gdr_list,
                'PAR_norm': par_norm,
                'GDR_norm': gdr_norm,
                'color': color
            }
            
        # 添加对角线参考线
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random Prediction')
        
        # 璁剧疆鍥惧舰灞炴€?- ROC椋庢牸
        plt.xlabel('PAR', fontsize=14, fontweight='bold')
        plt.ylabel('GDR', fontsize=14, fontweight='bold')
        plt.title('', fontsize=16, fontweight='bold')
        
        # 网格和图例
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.legend(fontsize=11, loc='lower right')
        
        # 璁剧疆鍧愭爣杞磋寖鍥村拰鍒诲害
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # 璁剧疆鍒诲害鏍囩涓虹櫨鍒嗘瘮
        from matplotlib.ticker import FuncFormatter
        def percent_formatter(x, pos):
            return f'{x*100:.0f}%'
        
        plt.gca().xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))
        
        # 璁剧疆鍒诲害闂撮殧
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        
        # 纭繚姣斾緥涓€鑷达紙姝ｆ柟褰級
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 淇濆瓨鍥惧儚
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"鉁?DEC鏇茬嚎鍥惧凡淇濆瓨鑷? {output_path}")
        
        # 淇濆瓨鏁版嵁鍒癑SON鏂囦欢
        data_path = output_path.replace('.png', '_data.json').replace('.jpg', '_data.json')
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(all_curves_data, f, ensure_ascii=False, indent=2)
        print(f"鉁?DEC鏇茬嚎鏁版嵁宸蹭繚瀛樿嚦: {data_path}")
        
        # 鏄剧ず鍥惧儚
        plt.show()

def print_header():
    """鎵撳嵃绋嬪簭鏍囬"""
    print("\n" + "="*50)
    print("馃幆 ROC椋庢牸DEC鍔ㄦ€佹晥鐜囨洸绾跨敓鎴愬櫒")
    print("="*50)
    print(f"馃搳 PAR鑼冨洿: {PAR_MIN}% - {PAR_MAX}% (姝ラ暱: {STEP}%)")
    print(f"馃搱 鍥惧舰椋庢牸: ROC鏇茬嚎椋庢牸 (0-1鍧愭爣)")
    print(f"馃搧 杈撳嚭鐩綍: {OUTPUT_DIR}/")

def get_available_models():
    """Auto-detect available model files."""
    label_plots_dir = Path("label_plots")
    
    if not label_plots_dir.exists():
        print("鉂?鏈壘鍒?label_plots 鐩綍!")
        return {}
    
    # 鎵弿鎵€鏈?dat鏂囦欢
    dat_files = list(label_plots_dir.glob("*.dat"))
    
    if not dat_files:
        print("鉂?label_plots 鐩綍涓嬫湭鎵惧埌浠讳綍 .dat 鏂囦欢!")
        return {}
    
    print(f"\n馃搳 鑷姩妫€娴嬪埌 {len(dat_files)} 涓ā鍨嬫枃浠?")
    available_models = {}
    
    # 动态生成模型列表
    for i, dat_file in enumerate(sorted(dat_files), 1):
        # 浠庢枃浠跺悕鎻愬彇妯″瀷鍚嶇О
        model_name = dat_file.stem  # 鍘绘帀鎵╁睍鍚?        
        # 使用映射后的显示名称
        display_name = get_model_display_name(normalize_model_key(model_name))
        
        available_models[str(i)] = {
            'name': display_name,
            'file': str(dat_file)
        }
        
        print(f"  {i}. {display_name}")
    
    # 娣诲姞all閫夐」
    if available_models:
        print(f"  {len(available_models) + 1}. 鍏ㄩ儴鍙敤妯″瀷")
    
    return available_models

def select_models(available_models):
    """閫夋嫨瑕佷娇鐢ㄧ殑妯″瀷"""
    if not available_models:
        print("\n鉂?娌℃湁鎵惧埌鍙敤鐨勬ā鍨嬫枃浠?")
        return []
    
    # 璁＄畻"鍏ㄩ儴妯″瀷"鐨勯€夐」鍙风爜
    all_models_option = str(len(available_models) + 1)
    
    print("\n馃幆 閫夋嫨瑕佸姣旂殑妯″瀷:")
    print("杈撳叆妯″瀷缂栧彿锛岀敤閫楀彿鍒嗛殧 (渚嬪: 1,2,3)")
    print(f"输入 {all_models_option} 选择全部可用模型")
    
    while True:
        choice = input("\n璇疯緭鍏ラ€夋嫨: ").strip()
        
        if choice == all_models_option:
            selected = list(available_models.keys())
            break
        elif choice:
            try:
                selected = [x.strip() for x in choice.split(',')]
                # 楠岃瘉閫夋嫨
                for sel in selected:
                    if sel not in available_models:
                        raise ValueError(f"鏃犳晥閫夋嫨: {sel}")
                break
            except ValueError as e:
                print(f"鉂?杈撳叆閿欒: {e}")
                print("璇烽噸鏂拌緭鍏?")
        else:
            print("鉂?璇疯緭鍏ユ湁鏁堥€夋嫨!")
    
    # 鏄剧ず閫夋嫨缁撴灉
    print(f"\n鉁?宸查€夋嫨 {len(selected)} 涓ā鍨?")
    for sel in selected:
        print(f"  - {available_models[sel]['name']}")
    
    return selected

def main():
    """Main entry point."""
    try:
        # 检查工作目录
        if not Path(DEFAULT_DEPOSIT_FILE).exists():
            print(f"未找到 {DEFAULT_DEPOSIT_FILE} 文件!")
            return
            
        # 妫€鏌abel_plots鐩綍鏄惁瀛樺湪
        if not Path("label_plots").exists():
            print("鉂?鏈壘鍒?label_plots 鐩綍!")
            return
        
        # 纭繚杈撳嚭鐩綍瀛樺湪
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        # 程序主流程
        print_header()
        
        # 鑾峰彇鍙敤妯″瀷
        available_models = get_available_models()
        
        if not available_models:
            print("璇风‘淇?label_plots/ 鐩綍涓嬫湁瀵瑰簲鐨?.dat 鏂囦欢")
            return
        
        # 閫夋嫨妯″瀷
        selected_models = select_models(available_models)
        if not selected_models:
            return
        
        # 鐢熸垚杈撳嚭鏂囦欢鍚?        # 涓轰簡鏂囦欢鍚嶇畝娲侊紝鎻愬彇涓昏妯″瀷绫诲瀷
        simplified_names = []
        for sel in selected_models:
            name = available_models[sel]['name']
            # 鎻愬彇鎷彿鍓嶇殑涓昏鍚嶇О
            if '(' in name:
                main_part = name.split('(')[0].strip()
            else:
                main_part = name
            # 鏇挎崲鐗规畩瀛楃
            safe_name = main_part.replace('-', '').replace(' ', '').replace('_', '')
            simplified_names.append(safe_name)
        
        # 濡傛灉閫夋嫨浜嗗叏閮ㄦā鍨嬶紝浣跨敤鐗规畩鍚嶇О
        if len(selected_models) == len(available_models):
            output_filename = f"DEC_All_Models_{len(selected_models)}.png"
        else:
            output_filename = f"DEC_{'_'.join(simplified_names)}.png"
        
        output_path = f"{OUTPUT_DIR}/{output_filename}"
        
        # 开始生成
        print("\n开始生成 DEC 曲线...")
        print("="*40)
        
        try:
            # 鍒濆鍖栫敓鎴愬櫒
            dec_gen = SimpleDECGenerator()
            
            # 鍔犺浇妯″瀷鏁版嵁
            print("\n馃搳 鍔犺浇妯″瀷鏁版嵁...")
            models_data = []
            
            for sel in selected_models:
                model_info = available_models[sel]
                print(f"  馃搧 鍔犺浇 {model_info['name']}...")
                
                model_data = dec_gen.load_model_predictions(
                    model_info['file'], 
                    model_info['name']
                )
                models_data.append(model_data)
            
            print(f"\n成功加载 {len(models_data)} 个模型")
            
            # 鐢熸垚DEC鏇茬嚎
            print("\n馃搱 鐢熸垚DEC鏇茬嚎...")
            dec_gen.plot_dec_curves(models_data, output_path)
            
            # 鎴愬姛瀹屾垚
            print("\n" + "馃帀"*15)
            print("DEC 曲线生成完成")
            print(f"馃搧 鏂囦欢宸蹭繚瀛? {output_path}")
            print("馃帀"*15)
            
        except Exception as e:
            print(f"\n鉂?鐢熸垚澶辫触: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n馃憢 绋嬪簭缁撴潫锛屾劅璋娇鐢紒")
        
    except KeyboardInterrupt:
        print("\n\n馃憢 绋嬪簭琚敤鎴蜂腑鏂紝鍐嶈!")
    except Exception as e:
        print(f"\n鉂?绋嬪簭閿欒: {e}")

if __name__ == "__main__":
    main()
