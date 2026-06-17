import h5py
import numpy as np
from scipy.interpolate import griddata
import os
import time

def load_magnetic_data(filename):
    """加载新格式的数据"""
    with h5py.File(filename, 'r') as f:
        coordinates = f['coordinates'][:]  # 每行是一个点的坐标
        vectors = f['vectors'][:]         # 每行是对应点的属性值
    
    print(f"输入数据维度:")
    print(f"- coordinates: {coordinates.shape}")
    print(f"- vectors: {vectors.shape}")
    
    # 分离坐标
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    
    # 找到唯一的x和y坐标值
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    
    # 创建规则网格
    x_grid, y_grid = np.meshgrid(unique_x, unique_y, indexing='ij')
    
    # 对每个属性分量进行插值
    n_features = vectors.shape[1]  # 获取特征维度
    magnetic_grid = np.zeros((len(unique_x), len(unique_y), n_features))
    for i in range(n_features):
        magnetic_grid[:,:,i] = griddata((x, y), vectors[:,i], 
                                      (x_grid, y_grid), 
                                      method='cubic')
    
    print(f"网格数据形状: {magnetic_grid.shape}")
    return x_grid, y_grid, magnetic_grid

def pad_data(data, window_size, pad_mode='reflect'):
    """
    对数据进行扩边处理
    
    参数:
    - data: 原始数据，形状为(rows, cols, n_features)
    - window_size: 窗口大小
    - pad_mode: 扩边模式，可选值包括
      'edge': 复制边缘值
      'constant': 常数填充
      'reflect': 镜像填充
      'symmetric': 对称填充
      'wrap': 循环填充
    
    返回:
    - padded_data: 扩边后的数据
    """
    # 计算需要的padding大小
    pad_size = window_size // 2
    rows, cols, n_features = data.shape
    
    print(f"原始数据形状: {data.shape}")
    print(f"扩边大小: {pad_size} (窗口大小的一半)")
    print(f"扩边模式: {pad_mode}")
    
    # 创建扩边后的数据数组
    padded_data = np.zeros((rows + 2*pad_size, cols + 2*pad_size, n_features))
    
    # 将原始数据放入中心位置
    padded_data[pad_size:pad_size+rows, pad_size:pad_size+cols, :] = data
    
    # 根据不同的填充模式进行扩边
    if pad_mode == 'edge':
        # 边缘填充（使用边缘值复制）
        # 填充上下边缘
        for i in range(pad_size):
            padded_data[i, pad_size:pad_size+cols, :] = data[0, :, :]  # 上边缘
            padded_data[pad_size+rows+i, pad_size:pad_size+cols, :] = data[-1, :, :]  # 下边缘
        
        # 填充左右边缘
        for i in range(pad_size):
            padded_data[:, i, :] = padded_data[:, pad_size, :]  # 左边缘
            padded_data[:, pad_size+cols+i, :] = padded_data[:, pad_size+cols-1, :]  # 右边缘
    
    elif pad_mode == 'constant':
        # 常数填充（使用0值填充）
        # 这里已经默认为0，不需要额外处理
        pass
    
    elif pad_mode == 'reflect':
        # 镜像填充
        for c in range(n_features):
            # 为每个特征通道分别处理
            channel_data = data[:, :, c]
            
            # 上边缘
            for i in range(pad_size):
                reflect_i = pad_size - i
                padded_data[i, pad_size:pad_size+cols, c] = channel_data[reflect_i, :]
            
            # 下边缘
            for i in range(pad_size):
                reflect_i = rows - 2 - i
                padded_data[pad_size+rows+i, pad_size:pad_size+cols, c] = channel_data[reflect_i, :]
            
            # 左边缘
            for j in range(pad_size):
                reflect_j = pad_size - j
                padded_data[:, j, c] = padded_data[:, reflect_j, c]
            
            # 右边缘
            for j in range(pad_size):
                reflect_j = cols + pad_size - 2 - j
                padded_data[:, pad_size+cols+j, c] = padded_data[:, reflect_j, c]
    
    elif pad_mode == 'symmetric':
        # 对称填充
        for c in range(n_features):
            # 为每个特征通道分别处理
            channel_data = data[:, :, c]
            
            # 上边缘
            for i in range(pad_size):
                symmetric_i = i
                padded_data[i, pad_size:pad_size+cols, c] = channel_data[symmetric_i, :]
            
            # 下边缘
            for i in range(pad_size):
                symmetric_i = rows - 1 - i
                padded_data[pad_size+rows+i, pad_size:pad_size+cols, c] = channel_data[symmetric_i, :]
            
            # 左右边缘（对整个填充后的数组）
            for j in range(pad_size):
                # 左边缘
                symmetric_j = j
                padded_data[:, j, c] = padded_data[:, pad_size + symmetric_j, c]
                
                # 右边缘
                symmetric_j = cols - 1 - j
                padded_data[:, pad_size+cols+j, c] = padded_data[:, pad_size + symmetric_j, c]
    
    elif pad_mode == 'wrap':
        # 循环填充
        for c in range(n_features):
            # 为每个特征通道分别处理
            channel_data = data[:, :, c]
            
            # 上边缘
            for i in range(pad_size):
                wrap_i = rows - pad_size + i
                padded_data[i, pad_size:pad_size+cols, c] = channel_data[wrap_i, :]
            
            # 下边缘
            for i in range(pad_size):
                wrap_i = i
                padded_data[pad_size+rows+i, pad_size:pad_size+cols, c] = channel_data[wrap_i, :]
            
            # 左右边缘（对整个填充后的数组）
            for j in range(pad_size):
                # 左边缘
                wrap_j = cols - pad_size + j
                padded_data[:, j, c] = padded_data[:, pad_size + wrap_j, c]
                
                # 右边缘
                wrap_j = j
                padded_data[:, pad_size+cols+j, c] = padded_data[:, pad_size + wrap_j, c]
    
    else:
        raise ValueError(f"不支持的填充模式: {pad_mode}")
    
    print(f"扩边后数据形状: {padded_data.shape}")
    return padded_data

def sliding_window(data, window_size=32, stride=2):
    """使用滑动窗口切割多维数据，返回窗口数据和位置信息"""
    windows = []
    positions = []
    
    # 获取数据维度
    rows, cols, n_features = data.shape
    print(f"处理数据维度: {n_features}")
    
    # 计算可以提取的窗口数量
    n_rows = rows - window_size + 1
    n_cols = cols - window_size + 1
    
    print(f"可提取窗口范围: 行(0-{n_rows-1}), 列(0-{n_cols-1})")
    print(f"补边后数据总大小: {rows}×{cols}")
    
    # 使用滑动窗口提取数据
    for i in range(0, n_rows, stride):
        for j in range(0, n_cols, stride):
            # 提取窗口数据
            window = data[i:i+window_size, j:j+window_size, :]
            
            # 只保存完整的窗口
            if window.shape == (window_size, window_size, n_features):
                windows.append(window)
                # 记录窗口中心位置
                center_i = i + window_size // 2
                center_j = j + window_size // 2
                positions.append((center_i, center_j))
    
    # 如果使用步长大于1，确保最后一行和最后一列被覆盖
    if stride > 1:
        # 检查是否已经包含最后一行和最后一列
        last_row_idx = rows - window_size
        last_col_idx = cols - window_size
        
        # 添加最后一行的窗口
        if last_row_idx % stride != 0:
            for j in range(0, n_cols, stride):
                if j + window_size <= cols:  # 确保窗口在数据范围内
                    window = data[last_row_idx:last_row_idx+window_size, j:j+window_size, :]
                    if window.shape == (window_size, window_size, n_features):
                        windows.append(window)
                        center_i = last_row_idx + window_size // 2
                        center_j = j + window_size // 2
                        positions.append((center_i, center_j))
        
        # 添加最后一列的窗口
        if last_col_idx % stride != 0:
            for i in range(0, n_rows, stride):
                if i + window_size <= rows:  # 确保窗口在数据范围内
                    window = data[i:i+window_size, last_col_idx:last_col_idx+window_size, :]
                    if window.shape == (window_size, window_size, n_features):
                        windows.append(window)
                        center_i = i + window_size // 2
                        center_j = last_col_idx + window_size // 2
                        positions.append((center_i, center_j))
        
        # 添加右下角窗口
        if last_row_idx % stride != 0 and last_col_idx % stride != 0:
            window = data[last_row_idx:last_row_idx+window_size, last_col_idx:last_col_idx+window_size, :]
            if window.shape == (window_size, window_size, n_features):
                windows.append(window)
                center_i = last_row_idx + window_size // 2
                center_j = last_col_idx + window_size // 2
                positions.append((center_i, center_j))
    
    print(f"提取的窗口数量: {len(windows)}个")
    
    # 转换数据格式
    windows = np.array(windows)  # 现在形状为(N, window_size, window_size, n_features)
    # 转置为(window_size, window_size, n_features, N)
    windows = np.transpose(windows, (1, 2, 3, 0))
    
    print(f"转换后的窗口数据形状: {windows.shape}")
    
    # 将位置信息转换为numpy数组
    positions = np.array(positions)
    
    # 验证窗口覆盖范围
    if len(positions) > 0:
        min_i, max_i = np.min(positions[:, 0]), np.max(positions[:, 0])
        min_j, max_j = np.min(positions[:, 1]), np.max(positions[:, 1])
        
        # 计算期望的窗口中心覆盖范围
        expected_min_i = window_size // 2
        expected_max_i = rows - window_size // 2 - 1
        expected_min_j = window_size // 2
        expected_max_j = cols - window_size // 2 - 1
        
        print(f"窗口中心索引实际范围: 行({min_i}-{max_i}), 列({min_j}-{max_j})")
        print(f"窗口中心索引期望范围: 行({expected_min_i}-{expected_max_i}), 列({expected_min_j}-{expected_max_j})")
        
        # 检查是否有缺失的覆盖区域
        if min_i > expected_min_i or max_i < expected_max_i or min_j > expected_min_j or max_j < expected_max_j:
            print("警告: 窗口覆盖范围不完整!")
    
    return windows, positions

def index_to_geo_coords(positions, x_offset=6015, y_offset=82532, x_max=6895, y_max=82884, rows=441, cols=177):
    """
    将索引坐标转换为地理坐标
    
    参数:
    - positions: 索引坐标数组，形状为(N, 2)，每行为(i, j)
    - x_offset: X轴地理坐标的起始值
    - y_offset: Y轴地理坐标的起始值
    - x_max: X轴地理坐标的最大值
    - y_max: Y轴地理坐标的最大值
    - rows: 数据的行数
    - cols: 数据的列数
    
    返回:
    - geo_positions: 地理坐标数组，形状为(N, 2)，每行为(x, y)
    """
    if positions.shape[1] != 2:
        raise ValueError("positions应当是包含(i,j)索引对的二维数组")
    
    # 创建地理坐标数组
    geo_positions = np.zeros_like(positions, dtype=float)
    
    # 自适应计算坐标转换的缩放因子
    x_scale = (x_max - x_offset) / (rows - 1) if rows > 1 else 1
    y_scale = (y_max - y_offset) / (cols - 1) if cols > 1 else 1
    
    # 应用线性变换
    geo_positions[:, 0] = x_offset + positions[:, 0] * x_scale
    geo_positions[:, 1] = y_offset + positions[:, 1] * y_scale
    
    return geo_positions

def save_windows_to_h5(filename, windows, positions, window_size=32, x_offset=6015, y_offset=82532, orig_shape=None):
    """保存窗口数据和地理坐标到h5文件"""
    max_retries = 3
    retry_count = 0
    
    # 获取原始数据形状
    if orig_shape is None:
        print("警告: 未提供原始数据形状，使用默认值(441, 177)")
        rows, cols = 441, 177
    else:
        rows, cols = orig_shape[:2]
    
    # 设置地理坐标范围
    x_max = 6895
    y_max = 82884
    
    # 计算缩放因子
    x_scale = (x_max - x_offset) / (rows - 1) if rows > 1 else 1
    y_scale = (y_max - y_offset) / (cols - 1) if cols > 1 else 1
    
    print(f"坐标转换参数:")
    print(f"- 原始数据形状: {rows}×{cols}")
    print(f"- X轴: 索引0-{rows-1}对应坐标{x_offset}-{x_max}, 缩放因子={x_scale:.6f}")
    print(f"- Y轴: 索引0-{cols-1}对应坐标{y_offset}-{y_max}, 缩放因子={y_scale:.6f}")
    
    # 转换为地理坐标
    geo_positions = index_to_geo_coords(positions, x_offset, y_offset, x_max, y_max, rows, cols)
    
    while retry_count < max_retries:
        try:
            # 如果文件存在，先尝试删除
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"删除已存在的文件: {filename}")
                except OSError:
                    print(f"无法删除已存在的文件: {filename}")
                    pass
            
            # 等待一小段时间
            time.sleep(1)
            
            # 尝试保存文件
            with h5py.File(filename, 'w') as f:
                f.create_dataset('windows', data=windows)
                f.create_dataset('positions', data=geo_positions)  # 保存地理坐标
                f.create_dataset('index_positions', data=positions)  # 也保存索引坐标以便参考
                f.attrs['window_size'] = window_size
                f.attrs['x_offset'] = x_offset
                f.attrs['y_offset'] = y_offset
                f.attrs['coordinate_type'] = 'geographic'
                # 保存坐标范围信息
                f.attrs['x_scale'] = x_scale
                f.attrs['y_scale'] = y_scale
                f.attrs['x_max'] = x_max
                f.attrs['y_max'] = y_max
                f.attrs['data_rows'] = rows
                f.attrs['data_cols'] = cols
            print(f"成功保存文件: {filename}")
            return True
            
        except OSError as e:
            retry_count += 1
            print(f"保存失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
            time.sleep(2)  # 等待2秒后重试
    
    raise OSError(f"无法保存文件 {filename} 经过 {max_retries} 次尝试")

def list_h5_files():
    """列出当前目录下的所有h5文件"""
    h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    if not h5_files:
        print("当前目录下没有找到.h5文件！")
        return None
    
    print("\n当前目录下的h5文件：")
    for i, file in enumerate(h5_files, 1):
        print(f"[{i}] {file}")
    return h5_files

def main():
    # 列出并选择输入文件
    h5_files = list_h5_files()
    if not h5_files:
        return
    
    while True:
        try:
            choice = int(input("\n请输入要处理的文件编号: "))
            if 1 <= choice <= len(h5_files):
                input_file = h5_files[choice-1]
                break
            else:
                print(f"请输入1到{len(h5_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    # 设置输出文件名
    default_output = f"windows_{os.path.splitext(input_file)[0]}.h5"
    output_file = input(f"请输入保存结果的文件名（直接回车使用默认名称：{default_output}）: ")
    if not output_file:
        output_file = default_output
    
    # 加载数据
    print(f"\n正在加载数据文件: {input_file}")
    x_grid, y_grid, magnetic_grid = load_magnetic_data(input_file)
    
    # 获取并输出数据的实际维度
    rows, cols = magnetic_grid.shape[:2]
    print(f"数据实际维度: {rows}×{cols}")
    
    # 设置地理坐标范围
    x_min, x_max = 6015, 6895
    y_min, y_max = 82532, 82884
    
    # 根据实际数据尺寸计算期望的地理坐标映射关系
    x_scale = (x_max - x_min) / (rows - 1) if rows > 1 else 1
    y_scale = (y_max - y_min) / (cols - 1) if cols > 1 else 1
    
    print(f"坐标映射: ")
    print(f"- X轴: 索引0-{rows-1}对应坐标{x_min}-{x_max}, 缩放因子={x_scale:.6f}")
    print(f"- Y轴: 索引0-{cols-1}对应坐标{y_min}-{y_max}, 缩放因子={y_scale:.6f}")
    
    # 设置滑动窗口参数
    while True:
        try:
            window_size = int(input("\n请输入滑动窗口大小（建议16或32）: "))
            if window_size > 0:
                break
            print("窗口大小必须大于0")
        except ValueError:
            print("请输入有效的数字")
    
    while True:
        try:
            stride = int(input("请输入滑动步长（建议1-4）: "))
            if stride > 0:
                break
            print("步长必须大于0")
        except ValueError:
            print("请输入有效的数字")
    
    # 选择扩边模式
    print("\n请选择扩边模式:")
    print("1. edge - 复制边缘值")
    print("2. constant - 常数填充（零值）")
    print("3. reflect - 镜像填充")
    print("4. symmetric - 对称填充")
    print("5. wrap - 循环填充")
    
    pad_modes = {
        1: 'edge',
        2: 'constant',
        3: 'reflect',
        4: 'symmetric',
        5: 'wrap'
    }
    
    while True:
        try:
            pad_choice = int(input("\n请输入扩边模式编号 [1]: "))
            if 1 <= pad_choice <= 5:
                pad_mode = pad_modes[pad_choice]
                break
            elif pad_choice == '':
                pad_mode = 'edge'  # 默认使用边缘填充
                break
            else:
                print(f"请输入1到5之间的数字")
        except ValueError:
            pad_mode = 'edge'  # 默认使用边缘填充
            break
    
    # 对数据进行扩边处理
    print("\n正在进行扩边处理...")
    padded_data = pad_data(magnetic_grid, window_size, pad_mode)
    
    # 切割数据
    print("\n正在进行数据切割...")
    windows, positions = sliding_window(padded_data, window_size, stride)
    
    # 将索引坐标映射回原始数据范围（减去padding偏移）
    pad_size = window_size // 2
    positions = positions - pad_size
    
    # 保存结果
    print(f"\n正在保存结果到: {output_file}")
    
    # 设置坐标转换的偏移值
    x_offset = x_min  # X轴方向：索引0对应坐标x_min
    y_offset = y_min  # Y轴方向：索引0对应坐标y_min
    
    # 将原始数据形状传递给保存函数
    save_windows_to_h5(output_file, windows, positions, window_size, x_offset, y_offset, magnetic_grid.shape)
    
    # 打印信息
    print(f"\n处理完成！")
    print(f"原始数据大小: {magnetic_grid.shape}")
    print(f"扩边后数据大小: {padded_data.shape}")
    print(f"窗口大小: {window_size}×{window_size}")
    print(f"步长: {stride}")
    print(f"提取的窗口数量: {windows.shape[3]}")
    print(f"窗口数据形状: {windows.shape}")
    
    # 验证窗口中心范围
    orig_rows, orig_cols = magnetic_grid.shape[:2]
    min_i, max_i = np.min(positions[:, 0]), np.max(positions[:, 0])
    min_j, max_j = np.min(positions[:, 1]), np.max(positions[:, 1])
    
    print("\n窗口中心索引位置范围:")
    print(f"行(i): {min_i} 到 {max_i} (原始数据范围: 0 到 {orig_rows-1})")
    print(f"列(j): {min_j} 到 {max_j} (原始数据范围: 0 到 {orig_cols-1})")
    
    # 计算并显示地理坐标范围
    # 使用自适应的坐标转换
    min_x = x_min + min_i * x_scale
    max_x = x_min + max_i * x_scale
    min_y = y_min + min_j * y_scale
    max_y = y_min + max_j * y_scale
    
    print("\n窗口中心地理坐标范围:")
    print(f"X: {min_x:.2f} 到 {max_x:.2f} (完整范围: {x_min} 到 {x_max})")
    print(f"Y: {min_y:.2f} 到 {max_y:.2f} (完整范围: {y_min} 到 {y_max})")
    
    # 验证数据
    with h5py.File(output_file, 'r') as f:
        print("\n保存的数据信息:")
        print("数据集列表:", list(f.keys()))
        print("窗口数据形状:", f['windows'].shape)
        print("位置数据形状:", f['positions'].shape)
        print("窗口大小:", f.attrs['window_size'])
        print("坐标类型:", f.attrs['coordinate_type'])
        print("X坐标偏移:", f.attrs['x_offset'])
        print("Y坐标偏移:", f.attrs['y_offset'])
        print("X坐标缩放因子:", f.attrs['x_scale'])
        print("Y坐标缩放因子:", f.attrs['y_scale'])
        print("原始数据形状:", (f.attrs['data_rows'], f.attrs['data_cols']))
        
        # 显示前几个窗口的位置信息
        geo_pos = f['positions'][:]
        idx_pos = f['index_positions'][:]
        print("\n前5个窗口的位置信息:")
        for i in range(min(5, len(geo_pos))):
            print(f"窗口 {i+1}:")
            print(f"  索引坐标: (i={idx_pos[i,0]}, j={idx_pos[i,1]})")
            print(f"  地理坐标: (X={geo_pos[i,0]:.2f}, Y={geo_pos[i,1]:.2f})")

if __name__ == "__main__":
    main() 