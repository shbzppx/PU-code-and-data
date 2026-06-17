import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm

def load_combined_data(filename):
    """加载合并后的H5数据"""
    with h5py.File(filename, 'r') as f:
        # 检查必要的数据集是否存在
        if 'coordinates' not in f or 'vectors' not in f:
            raise ValueError("文件格式不正确：缺少coordinates或vectors数据集")
        
        coordinates = f['coordinates'][:]
        vectors = f['vectors'][:]
        
        # 获取原始文件名列表（如果存在）
        file_names = f.attrs.get('file_names', [])
        
    print(f"\n数据集信息：")
    print(f"坐标点数量: {len(coordinates)}")
    print(f"特征维度数: {vectors.shape[1]}")
    
    if len(file_names) > 0:
        print("\n特征来源文件：")
        for i, name in enumerate(file_names, 1):
            print(f"{i}. {name}")
    
    return coordinates, vectors, file_names

def save_to_dat(filename, coordinates, vectors, header=None):
    """
    保存数据到DAT文件
    
    参数:
    - filename: 输出文件名
    - coordinates: 坐标数据 (N x 2)
    - vectors: 特征数据 (N x M)
    - header: 表头字符串（可选）
    """
    try:
        with open(filename, 'w') as f:
            # 写入表头
            if header is None:
                # 生成默认表头
                feature_cols = [f"Feature_{i+1}" for i in range(vectors.shape[1])]
                header = "X Y " + " ".join(feature_cols)
            f.write(f"{header}\n")
            
            # 写入数据
            print("\n正在写入数据...")
            for (x, y), vector in tqdm(zip(coordinates, vectors), total=len(coordinates)):
                # 将所有值格式化为字符串
                values = [f"{x:.6f}", f"{y:.6f}"] + [f"{v:.6f}" for v in vector]
                # 写入一行数据
                f.write(" ".join(values) + "\n")
        
        print(f"\n数据已保存到: {filename}")
        
        # 计算文件大小
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"文件大小: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"保存文件时出错: {str(e)}")
        raise

def list_h5_files():
    """列出当前目录下的所有h5文件"""
    h5_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    if not h5_files:
        print("当前目录下没有找到H5文件！")
        return None
    
    print("\n当前目录下的H5文件：")
    for i, file in enumerate(h5_files, 1):
        print(f"[{i}] {file}")
    return h5_files

def main():
    parser = argparse.ArgumentParser(description='合并H5数据导出工具')
    parser.add_argument('-i', '--input', help='输入的合并H5文件')
    parser.add_argument('-o', '--output', help='输出的DAT文件')
    parser.add_argument('--header', help='自定义表头（用空格分隔的列名）')
    args = parser.parse_args()
    
    try:
        # 获取输入文件
        input_file = args.input
        
        if input_file is None:
            # 列出可用文件供选择
            h5_files = list_h5_files()
            if not h5_files:
                return
            
            while True:
                try:
                    choice = int(input("\n请选择输入文件 (输入序号): "))
                    if 1 <= choice <= len(h5_files):
                        input_file = h5_files[choice-1]
                        break
                    else:
                        print(f"请输入1到{len(h5_files)}之间的数字")
                except ValueError:
                    print("请输入有效的数字")
        
        # 设置输出文件名
        output_file = args.output
        if output_file is None:
            # 默认输出文件名：替换扩展名为.dat
            output_file = os.path.splitext(input_file)[0] + '.dat'
        
        # 加载数据
        print(f"\n正在加载数据: {input_file}")
        coordinates, vectors, file_names = load_combined_data(input_file)
        
        # 准备表头
        header = args.header
        if header is None and len(file_names) > 0:
            # 如果有原始文件名，使用它们作为特征名
            feature_names = []
            for name in file_names:
                base_name = os.path.splitext(name)[0]
                if vectors.shape[1] == len(file_names):
                    # 每个文件贡献一个特征
                    feature_names.append(base_name)
                else:
                    # 每个文件可能贡献多个特征
                    feature_names.extend([f"{base_name}_{i+1}" for i in range(vectors.shape[1]//len(file_names))])
            header = "X Y " + " ".join(feature_names)
        
        # 保存数据
        print(f"\n正在保存数据到: {output_file}")
        save_to_dat(output_file, coordinates, vectors, header)
        
        # 打印统计信息
        print(f"\n导出完成:")
        print(f"总点数: {len(coordinates)}")
        print(f"特征数: {vectors.shape[1]}")
        
        # 显示数据预览
        print("\n数据预览 (前3行):")
        with open(output_file, 'r') as f:
            for i, line in enumerate(f):
                if i > 3:
                    break
                print(line.strip())
        
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 