import h5py
import numpy as np
from pathlib import Path

def list_and_select_h5_files():
    """列出并选择要合并的h5文件"""
    h5_files = list(Path('.').glob('*.h5'))
    if not h5_files:
        print("当前目录下没有找到h5文件")
        return None
    
    print("\n找到的h5文件：")
    for i, file in enumerate(h5_files, 1):
        print(f"{i}. {file}")
    
    selected_files = []
    while True:
        try:
            choice = input("\n请输入要合并的文件编号 (用空格分隔，直接按回车结束选择): ").strip()
            if choice == "":
                if not selected_files:
                    print("请至少选择一个文件")
                    continue
                break
            
            
            indices = [int(x) for x in choice.split()]
            for idx in indices:
                if 1 <= idx <= len(h5_files) and h5_files[idx-1] not in selected_files:
                    selected_files.append(h5_files[idx-1])
                    print(f"已选择: {h5_files[idx-1]}")
                else:
                    print(f"无效的选择: {idx}")
        except ValueError:
            print("请输入有效的数字")
    
    print("\n已选择的文件：")
    for i, file in enumerate(selected_files, 1):
        print(f"{i}. {file}")
    
    return selected_files

def combine_h5_data(input_files, output_file):
    """合并多个h5文件的数据，保持coordinates不变，合并vectors的维度"""
    # 首先从第一个文件获取网格坐标和vectors的维度信息
    with h5py.File(input_files[0], 'r') as f:
        # 检查必要的数据集是否存在
        if 'coordinates' in f and 'vectors' in f:
            # 已经是合并格式的文件
            coordinates = f['coordinates'][:]
            num_points = len(coordinates)
            vector_dims = f['vectors'].shape[1]
            print("\n检测到合并格式的文件")
        elif 'x_coords' in f and 'y_coords' in f and 'data' in f:
            # 原始格式的文件
            x_coords = f['x_coords'][:]
            y_coords = f['y_coords'][:]
            # X, Y = np.meshgrid(x_coords, y_coords)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')  # 添加 indexing='ij' 参数

            coordinates = np.column_stack((X.flatten(), Y.flatten()))
            num_points = len(coordinates)
            vector_dims = 1  # 原始文件每个点只有一个数据值
            print("\n检测到原始格式的文件")
        else:
            print("错误：文件格式不正确")
            return
        
        print("\n数据集信息：")
        print(f"坐标点数量: {num_points}")
        print(f"每个点的向量维度: {vector_dims}")
        print("\n前5个点的坐标：")
        for i in range(min(5, num_points)):
            print(f"点 {i+1}: ({coordinates[i][0]}, {coordinates[i][1]})")
    
    # 计算合并后的vectors总维度
    total_dims = 0
    for file_path in input_files:
        with h5py.File(file_path, 'r') as f:
            if 'vectors' in f:
                total_dims += f['vectors'].shape[1]
            elif 'data' in f:
                total_dims += 1
            else:
                print(f"错误：文件 {file_path} 格式不正确")
                return
    
    print(f"\n合并后的总维度: {total_dims}")
    
    # 创建输出文件
    with h5py.File(output_file, 'w') as f_out:
        # 创建coordinates数据集 (N x 2)
        coords_dataset = f_out.create_dataset('coordinates', data=coordinates, dtype=np.float32)
        
        # 创建vectors数据集 (N x M)，其中M是所有维度之和
        vectors = f_out.create_dataset('vectors', shape=(num_points, total_dims), dtype=np.float32)
        
        # 逐个读取文件的数据集
        current_dim = 0
        for i, file_path in enumerate(input_files):
            print(f"\n处理文件 {i+1}/{len(input_files)}: {file_path}")
            with h5py.File(file_path, 'r') as f_in:
                # 验证坐标点是否相同
                if 'coordinates' in f_in:
                    if not np.allclose(f_in['coordinates'][:], coordinates):
                        print(f"警告：文件 {file_path} 的坐标点与第一个文件不完全相同")
                    file_vectors = f_in['vectors'][:]
                    dims = file_vectors.shape[1]
                    print(f"文件vectors形状: {file_vectors.shape}")
                else:
                    # 原始格式文件
                    # x_in = f_in['x_coords'][:]
                    # y_in = f_in['y_coords'][:]
                    # X_in, Y_in = np.meshgrid(x_in, y_in)
                    # coords_in = np.column_stack((X_in.flatten(), Y_in.flatten()))
                    x_in = f_in['x_coords'][:]
                    y_in = f_in['y_coords'][:]
                    X_in, Y_in = np.meshgrid(x_in, y_in, indexing='ij')  # 添加 indexing='ij' 参数
                    coords_in = np.column_stack((X_in.flatten(), Y_in.flatten()))

                    if not np.allclose(coords_in, coordinates):
                        print(f"警告：文件 {file_path} 的坐标点与第一个文件不完全相同")
                    file_vectors = f_in['data'][:].flatten()[:, np.newaxis]  # 转换为2D数组
                    dims = 1
                
                if len(file_vectors) != num_points:
                    print(f"错误：文件 {file_path} 中的数据点数量 ({len(file_vectors)}) 与坐标点数量 ({num_points}) 不匹配")
                    return
                
                # 将当前文件的数据复制到对应的位置
                print(f"复制数据到维度 {current_dim} 到 {current_dim+dims}")
                vectors[:, current_dim:current_dim+dims] = file_vectors
                current_dim += dims
                
                # 打印前几个点的信息用于验证
                print(f"前5个点的向量值：")
                for j in range(min(5, num_points)):
                    if dims > 3:
                        print(f"点 {j+1}: {file_vectors[j,:3]}... (显示前3个维度)")
                    else:
                        print(f"点 {j+1}: {file_vectors[j]}")
        
        # 记录文件名
        file_names = [str(file.name) for file in input_files]
        f_out.attrs['file_names'] = file_names
        
        print("\n输出文件信息：")
        print(f"坐标数据形状: {coords_dataset.shape}")
        print(f"属性值数据形状: {vectors.shape}")
        
        # 验证最终结果
        print("\n验证前5个点的最终向量值：")
        for i in range(min(5, num_points)):
            if total_dims > 3:
                print(f"点 {i+1}: {vectors[i,:3]}... (显示前3个维度)")
            else:
                print(f"点 {i+1}: {vectors[i]}")

def main():
    # 获取用户选择的h5文件
    input_files = list_and_select_h5_files()
    if not input_files:
        return
    
    # 生成输出文件名
    output_file = "combined_data.h5"
    
    # 合并数据
    print(f"\n开始合并数据到: {output_file}")
    combine_h5_data(input_files, output_file)
    
    # 验证结果
    with h5py.File(output_file, 'r') as f:
        print("\n合并完成！")
        print("坐标数据形状:", f['coordinates'].shape)
        print("属性值数据形状:", f['vectors'].shape)
        print("\n包含的文件:")
        for i, name in enumerate(f.attrs['file_names'], 1):
            print(f"{i}. {name}")

if __name__ == "__main__":
    main()