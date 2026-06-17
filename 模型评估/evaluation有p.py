import torch

def evaluate_model(model, data_loader, loss_func, device):
    """评估模型的损失"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output.view(-1), target)
            total_loss += loss.item() * len(data)
    return total_loss / len(data_loader.dataset)

def compute_error(model, data_loader, device):
    """计算错误率"""
    model.eval()
    total_error = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            error = model.error(data, target)
            total_error += error * len(data)
            total_samples += len(data)
    return total_error / total_samples

def compute_pos_recall(model, data_loader, device):
    """计算正样本召回率：预测为正的真正样本数量 / 总正样本数量"""
    model.eval()
    total_true_positives = 0  # 预测正确的正样本数
    total_positives = 0  # 总的正样本数
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # 获取预测结果
            output = model(data)
            predictions = (output > 0).float()  # 正值为正类
            
            # 找出真正的正样本
            positives = (target == 1)
            true_positives = (predictions.view(-1) == 1) & positives
            
            # 更新计数
            total_positives += positives.sum().item()
            total_true_positives += true_positives.sum().item()
    
    # 避免除零错误
    if total_positives == 0:
        return 0.0
    
    return total_true_positives / total_positives 

def compute_r_squared_over_pr(model, data_loader, device):
    """计算评价指标 r²/Pr[f(X)=1]
    
    r: 正样本召回率
    Pr[f(X)=1]: 预测为正类的概率
    
    该指标同时考虑了模型查找正样本的能力和预测为正类的倾向性
    """
    model.eval()
    total_predictions = 0  # 总预测数
    total_positive_preds = 0  # 预测为正的数量
    total_true_positives = 0  # 预测正确的正样本数
    total_positives = 0  # 总的正样本数
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # 获取预测结果
            output = model(data)
            predictions = (output > 0).float()  # 正值为正类
            
            # 找出真正的正样本
            positives = (target == 1)
            true_positives = (predictions.view(-1) == 1) & positives
            
            # 更新计数
            batch_size = len(data)
            total_predictions += batch_size
            total_positive_preds += (predictions == 1).sum().item()
            total_positives += positives.sum().item()
            total_true_positives += true_positives.sum().item()
    
    # 计算召回率 r
    recall = 0.0
    if total_positives > 0:
        recall = total_true_positives / total_positives
    
    # 计算 Pr[f(X)=1]
    pr_positive = 0.0
    if total_predictions > 0:
        pr_positive = total_positive_preds / total_predictions
    
    # 计算 r²/Pr[f(X)=1]
    result = 0.0
    if pr_positive > 0:
        result = (recall ** 2) / pr_positive
    
    return {
        'recall': recall,
        'pr_positive': pr_positive,
        'r_squared_over_pr': result
    } 