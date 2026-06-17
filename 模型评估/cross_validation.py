import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import json

from utils import select_model, create_loss_function
from train_utils import train_model
from evaluation有p import compute_error, compute_pos_recall, evaluate_model
from visualization import plot_cv_curves

def prepare_cv_folds(X_train, y_train, n_folds, min_batch_size, stratified=True):
    """准备交叉验证的折"""
    # 获取所有样本索引
    indices = np.arange(len(X_train))
    
    if stratified:
        print("使用分层K折交叉验证...")
        # 获取所有标签
        all_labels = y_train
        
        # 分别获取正样本和负样本的索引
        pos_indices = indices[all_labels == 1]
        neg_indices = indices[all_labels == -1]
        
        # 确保每个折中的正负样本比例大致相同
        pos_fold_sizes = np.array_split(pos_indices, n_folds)
        neg_fold_sizes = np.array_split(neg_indices, n_folds)
        
        # 创建自定义的折，确保每个折的样本数能被批次大小整除
        fold_indices = []
        for fold_idx in range(n_folds):
            # 合并当前折的正负样本
            current_fold = np.concatenate([pos_fold_sizes[fold_idx], neg_fold_sizes[fold_idx]])
            # 打乱当前折内的样本顺序
            np.random.shuffle(current_fold)
            
            # 确保当前折的样本数是批次大小的整数倍
            remainder = len(current_fold) % min_batch_size
            if remainder > 0:
                # 调整折的大小，使其能被批次大小整除
                current_fold = current_fold[:-remainder]
            
            # 其他折作为训练集
            other_folds = []
            for other_idx in range(n_folds):
                if other_idx != fold_idx:
                    other_folds.append(np.concatenate([pos_fold_sizes[other_idx], neg_fold_sizes[other_idx]]))
            train_fold = np.concatenate(other_folds)
            
            # 确保训练集大小是批次大小的整数倍
            train_remainder = len(train_fold) % min_batch_size
            if train_remainder > 0:
                train_fold = train_fold[:-train_remainder]
                
            # 添加到fold_indices列表
            fold_indices.append((train_fold, current_fold))
            
            print(f"折 {fold_idx+1}: 训练集 {len(train_fold)} 样本, 验证集 {len(current_fold)} 样本")
    else:
        print("使用标准K折交叉验证...")
        # 打乱所有索引
        np.random.shuffle(indices)
        
        # 均匀划分为K个大小相近的折
        fold_size = len(indices) // n_folds
        adjusted_fold_size = (fold_size // min_batch_size) * min_batch_size  # 确保能被批次大小整除
        
        fold_indices = []
        for fold_idx in range(n_folds):
            start_idx = fold_idx * adjusted_fold_size
            end_idx = min((fold_idx + 1) * adjusted_fold_size, len(indices))
            
            if end_idx - start_idx < min_batch_size:
                # 如果这个折太小，就跳过
                continue
                
            val_fold = indices[start_idx:end_idx]
            train_fold = np.concatenate([indices[:start_idx], indices[end_idx:]])
            
            # 确保训练集大小是批次大小的整数倍
            train_remainder = len(train_fold) % min_batch_size
            if train_remainder > 0:
                train_fold = train_fold[:-train_remainder]
                
            fold_indices.append((train_fold, val_fold))
            
            print(f"折 {fold_idx+1}: 训练集 {len(train_fold)} 样本, 验证集 {len(val_fold)} 样本")
            
    return fold_indices

def train_with_cv(X_train, y_train, X_test, y_test, prior, args, model_dir, device, stop_token=None, progress_callback=None):
    """使用交叉验证的训练流程"""
    print(f"使用{args.cv_folds}折交叉验证...")
    
    # 创建完整的测试数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=True)
    
    # 创建完整数据集
    full_dataset = TensorDataset(X_train, y_train)
    
    # 非神经网络模型不使用交叉验证
    if args.model in ['ocsvm', '2step']:
        from train_utils import train_non_neural_model
        return train_non_neural_model(X_train, y_train, X_test, y_test, prior, args, model_dir, device)
    
    # 确保每个批次的大小一致，且至少有2个样本
    min_batch_size = max(2, args.batchsize)
    
    # 存储每折的性能指标
    cv_train_losses, cv_val_losses = [], []
    cv_train_errors, cv_val_errors = [], []
    cv_test_losses, cv_test_errors = [], []
    cv_train_recalls, cv_val_recalls, cv_test_recalls = [], [], []
    
    # 存储最佳模型的状态
    best_val_loss = float('inf')
    best_model_state = None
    
    # 准备交叉验证的折
    fold_indices = prepare_cv_folds(X_train, y_train, args.cv_folds, min_batch_size, 
                                    not args.no_stratified if hasattr(args, 'no_stratified') else args.stratified)
    
    # 执行交叉验证
    stop_requested = False
    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        if stop_token is not None and stop_token.is_set():
            print("检测到停止信号，停止后续折的训练。")
            stop_requested = True
            break
        
        print(f"\n开始训练第 {fold_idx+1}/{len(fold_indices)} 折...")
        
        # 创建训练和验证数据加载器
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_dataset, 
            batch_size=args.batchsize, 
            sampler=train_sampler
        )
        val_loader = DataLoader(
            full_dataset, 
            batch_size=args.batchsize, 
            sampler=val_sampler
        )
        
        # 确认数据加载器是否正确
        print(f"训练数据加载器: {len(train_loader)} 批次，大约 {len(train_loader)*args.batchsize} 样本")
        print(f"验证数据加载器: {len(val_loader)} 批次，大约 {len(val_loader)*args.batchsize} 样本")
        
        # 创建模型
        model = select_model(args.model)(prior, X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
        model = model.to(device)
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=args.stepsize)
        
        # 创建损失函数
        loss_func = create_loss_function(args, prior)
        
        # 初始化此折的指标记录
        fold_train_losses, fold_val_losses = [], []
        fold_train_errors, fold_val_errors = [], []
        fold_train_recalls, fold_val_recalls = [], []
        
        # 用于早停
        best_epoch = 0
        best_fold_val_loss = float('inf')
        best_fold_model_state = None
        patience_counter = 0
        
        # 决定是否使用早停
        use_early_stopping = args.early_stopping and not (hasattr(args, 'no_early_stopping') and args.no_early_stopping)
        
        # 训练循环
        for epoch in tqdm(range(args.epoch)):
            if stop_token is not None and stop_token.is_set():
                print("检测到停止信号，提前结束当前折训练。")
                stop_requested = True
                break
            
            # 训练阶段
            train_loss = train_model(model, train_loader, optimizer, loss_func, device, epoch)
            train_error = compute_error(model, train_loader, device)
            train_recall = compute_pos_recall(model, train_loader, device)
            
            # 验证阶段
            val_loss = evaluate_model(model, val_loader, loss_func, device)
            val_error = compute_error(model, val_loader, device)
            val_recall = compute_pos_recall(model, val_loader, device)
            
            # 记录结果
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            fold_train_errors.append(train_error)
            fold_val_errors.append(val_error)
            fold_train_recalls.append(train_recall)
            fold_val_recalls.append(val_recall)

            if progress_callback is not None:
                progress_callback({
                    'mode': 'cv',
                    'fold': fold_idx + 1,
                    'epoch': epoch + 1,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss)
                })

            # 打印结果（包含召回率）
            print(f'周期 {epoch+1}/{args.epoch}: 训练损失: {train_loss:.4f}, 错误率: {train_error:.4f}, 召回率: {train_recall:.4f}')
            print(f'验证损失: {val_loss:.4f}, 错误率: {val_error:.4f}, 召回率: {val_recall:.4f}')
            
            # 检查是否是最佳验证损失
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                best_fold_model_state = model.state_dict().copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停检查
            if use_early_stopping and patience_counter >= args.patience:
                print(f"早停触发！{args.patience} 个周期没有改善。")
                print(f"最佳验证损失出现在周期 {best_epoch+1}: {best_fold_val_loss:.4f}")
                break
        
        if fold_train_losses:
            # 加载最佳模型状态
            model.load_state_dict(best_fold_model_state)
            
            # 在测试集上评估
            test_loss = evaluate_model(model, test_loader, loss_func, device)
            test_error = compute_error(model, test_loader, device)
            test_recall = compute_pos_recall(model, test_loader, device)
            print(f'第 {fold_idx+1} 折测试结果 - 损失: {test_loss:.4f}, 错误率: {test_error:.4f}, 召回率: {test_recall:.4f}')
            
            # 保存此折的结果
            cv_train_losses.append(fold_train_losses)
            cv_val_losses.append(fold_val_losses)
            cv_train_errors.append(fold_train_errors)
            cv_val_errors.append(fold_val_errors)
            cv_test_losses.append(test_loss)
            cv_test_errors.append(test_error)
            cv_train_recalls.append(fold_train_recalls)
            cv_val_recalls.append(fold_val_recalls)
            cv_test_recalls.append(test_recall)
            
            # 保存模型权重
            fold_model_path = os.path.join(model_dir, f'model_fold{fold_idx+1}.pth')
            torch.save(model.state_dict(), fold_model_path)
            
            # 检查是否是整体最佳模型
            if best_fold_val_loss < best_val_loss:
                best_val_loss = best_fold_val_loss
                best_model_state = best_fold_model_state.copy()

        if stop_requested:
            break
    
    if not cv_test_losses:
        print("未完成任何交叉验证折，训练已停止。")
        return None

    # 计算交叉验证的平均性能
    avg_test_loss = np.mean(cv_test_losses)
    avg_test_error = np.mean(cv_test_errors)
    avg_test_recall = np.mean(cv_test_recalls)
    
    print("\n交叉验证结果汇总:")
    print(f"平均测试损失: {avg_test_loss:.4f} ± {np.std(cv_test_losses):.4f}")
    print(f"平均测试错误率: {avg_test_error:.4f} ± {np.std(cv_test_errors):.4f}")
    print(f"平均测试召回率: {avg_test_recall:.4f} ± {np.std(cv_test_recalls):.4f}")
    
    # 保存最佳模型
    best_model = select_model(args.model)(prior, X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
    best_model.load_state_dict(best_model_state)
    best_model = best_model.to(device)
    torch.save(best_model_state, os.path.join(model_dir, 'best_model.pth'))
    
    # 绘制交叉验证的学习曲线
    plot_cv_curves(
        cv_val_losses, cv_val_errors, cv_val_recalls,
        cv_test_losses, cv_test_errors, cv_test_recalls,
        avg_test_loss, avg_test_error, avg_test_recall,
        os.path.join(model_dir, 'cv_learning_curves.png'),
        len(cv_train_losses)
    )
    
    # 保存交叉验证结果
    cv_results = {
        'train_losses': cv_train_losses,
        'val_losses': cv_val_losses,
        'train_errors': cv_train_errors,
        'val_errors': cv_val_errors,
        'test_losses': cv_test_losses,
        'test_errors': cv_test_errors,
        'avg_test_loss': float(avg_test_loss),
        'avg_test_error': float(avg_test_error),
        'train_recalls': cv_train_recalls,
        'val_recalls': cv_val_recalls,
        'test_recalls': cv_test_recalls,
        'avg_test_recall': float(avg_test_recall)
    }
    
    with open(os.path.join(model_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=4)
    
    return best_model