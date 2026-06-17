import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def interpolate_deposit(deposit_file, ag_file, output_file):
    """
    将deposit.dat文件重新插值转换为与Ag.dat相同的格式
    
    参数:
    - deposit_file: 原始deposit.dat文件路径
    - ag_file: 参考Ag.dat文件路径
    - output_file: 输出文件路径
    """
    
    # 读取deposit.dat文件，确保X和Y列为浮点数类型
    try:
        # 尝试先读取一行检查是否有表头
        with open(deposit_file, 'r') as f:
            first_line = f.readline().strip().split()
            has_header = any(col.lower() in ['x', 'y'] for col in first_line)
        
        deposit_data = pd.read_csv(deposit_file, 
                                  sep='\s+', 
                                  names=['X', 'Y'] if not has_header else None,
                                  header=0 if has_header else None,
                                  dtype={'X': float, 'Y': float})
    except Exception as e:
        print(f"读取deposit.dat文件时出错: {str(e)}")
        raise
    
    # 读取Ag.dat文件获取目标网格点，确保坐标为浮点数类型
    try:
        # 尝试先读取一行检查是否有表头
        with open(ag_file, 'r') as f:
            first_line = f.readline().strip().split()
            has_header = any(col.lower() in ['x', 'y', 'value'] for col in first_line)
        
        ag_data = pd.read_csv(ag_file,
                             sep='\s+',
                             names=['X', 'Y', 'Value'] if not has_header else None,
                             header=0 if has_header else None,
                             dtype={'X': float, 'Y': float, 'Value': float})
    except Exception as e:
        print(f"读取Ag.dat文件时出错: {str(e)}")
        raise
    target_points = ag_data[['X', 'Y']].values
    
    # 创建标记数组，初始值为-1（非矿点）
    labels = np.full(len(target_points), -1)
    
    # 对每个矿点，找到最近的网格点并标记为1
    for _, row in deposit_data.iterrows():
        x, y = row['X'], row['Y']
        # 计算当前矿点到所有网格点的距离
        distances = np.sqrt(np.square(target_points[:,0] - x) + np.square(target_points[:,1] - y))
        # 找到距离最小的网格点索引
        nearest_point_idx = np.argmin(distances)
        # 将最近的网格点标记为矿点
        labels[nearest_point_idx] = 1
    
    # 创建输出DataFrame
    output_df = pd.DataFrame({
        'X': target_points[:, 0],
        'Y': target_points[:, 1],
        'Label': labels.astype(int)
    })
    
    # 保存结果
    output_df.to_csv(output_file, sep=' ', index=False, header=False)
    print(f"插值结果已保存到: {output_file}")

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 选择deposit.dat文件
    print("请选择deposit.dat文件...")
    deposit_file = filedialog.askopenfilename(title="选择deposit.dat文件",
                                            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
    if not deposit_file:
        print("未选择deposit.dat文件，程序退出")
        exit()
    
    # 选择Ag.dat文件
    print("请选择Ag.dat文件...")
    ag_file = filedialog.askopenfilename(title="选择Ag.dat文件",
                                       filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
    if not ag_file:
        print("未选择Ag.dat文件，程序退出")
        exit()
    
    # 选择保存位置
    print("请选择保存位置...")
    output_file = filedialog.asksaveasfilename(title="保存结果文件",
                                             defaultextension=".dat",
                                             filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
    if not output_file:
        print("未选择保存位置，程序退出")
        exit()
    
    # 执行插值转换
    interpolate_deposit(deposit_file, ag_file, output_file)