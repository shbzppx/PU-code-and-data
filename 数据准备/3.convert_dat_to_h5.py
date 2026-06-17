import numpy as np
import h5py
import pandas as pd
from pathlib import Path

def convert_dat_to_h5(input_file, output_file, grid_shape=(881, 353)):
    """
    将dat文件转换为h5格式，并重组数据为指定形状的网格
    
    参数:
    input_file: dat文件路径
    output_file: 输出的h5文件路径
    grid_shape: 目标网格形状，默认为(881, 353)
    """
    # 读取dat文件
    data = pd.read_csv(input_file, delimiter='\s+', header=None, names=['X', 'Y', 'Value'])
    
    # 获取唯一的X和Y坐标
    x_coords = sorted(data['X'].unique())
    y_coords = sorted(data['Y'].unique())
    
    # 确保坐标数量符合预期的网格形状
    if len(x_coords) != grid_shape[0] or len(y_coords) != grid_shape[1]:
        print(f"警告：实际坐标数量 ({len(x_coords)}, {len(y_coords)}) "
              f"与预期网格形状 {grid_shape} 不匹配")
    
    # 创建空的网格数组
    grid_data = np.zeros(grid_shape)
    
    # 创建坐标到索引的映射
    x_map = {x: i for i, x in enumerate(x_coords)}
    y_map = {y: i for i, y in enumerate(y_coords)}
    
    # 填充网格数据
    for _, row in data.iterrows():
        i = x_map[row['X']]
        j = y_map[row['Y']]
        grid_data[i, j] = row['Value']
    
    # 保存为h5文件
    with h5py.File(output_file, 'w') as f:
        # 创建数据集
        f.create_dataset('data', data=grid_data)
        # 保存坐标信息
        f.create_dataset('x_coords', data=x_coords)
        f.create_dataset('y_coords', data=y_coords)

def main():
    # 获取当前目录下的所有dat文件
    input_dir = Path('.')
    dat_files = list(input_dir.glob('*.dat'))
    
    if not dat_files:
        print("当前目录下没有找到dat文件")
        return
    
    # 显示文件列表供用户选择
    print("\n可用的dat文件：")
    for i, file in enumerate(dat_files, 1):
        print(f"{i}. {file}")
    
    # 获取用户选择
    while True:
        try:
            choice = int(input("\n请选择要转换的文件编号 (1-{}): ".format(len(dat_files))))
            if 1 <= choice <= len(dat_files):
                break
            print("无效的选择，请重试")
        except ValueError:
            print("请输入有效的数字")
    
    # 获取选中的文件
    dat_file = dat_files[choice - 1]
    
    # 直接在同一目录下生成h5文件
    output_file = dat_file.with_suffix('.h5')
    print(f"\n正在处理: {dat_file}")
    try:
        convert_dat_to_h5(dat_file, output_file)
        print(f"转换完成: {output_file}")
    except Exception as e:
        print(f"处理文件 {dat_file} 时出错: {str(e)}")

if __name__ == '__main__':
    main() 