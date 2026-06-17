import importlib.util
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
PARENT_MODEL_INIT = os.path.join(PARENT_DIR, "model", "__init__.py")

_parent_model = None
if os.path.exists(PARENT_MODEL_INIT):
    spec = importlib.util.spec_from_file_location(
        "_eval_parent_model_package", PARENT_MODEL_INIT
    )
    _parent_model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = _parent_model
    spec.loader.exec_module(_parent_model)
    __path__ = [os.path.join(PARENT_DIR, "model")]
    for _submodule in (
        "random_forest",
        "pu_random_forest",
        "one_class_svm",
        "two_step_pu",
        "linear",
        "mlp",
        "cnn",
        "cnn_transformer",
        "cnn_token_transformer",
    ):
        _loaded_name = f"{spec.name}.{_submodule}"
        if _loaded_name in sys.modules:
            sys.modules.setdefault(f"model.{_submodule}", sys.modules[_loaded_name])

if _parent_model is not None:
    LinearClassifier = _parent_model.LinearClassifier
    ThreeLayerPerceptron = _parent_model.ThreeLayerPerceptron
    MultiLayerPerceptron = _parent_model.MultiLayerPerceptron
    CNNTokenTransformer = _parent_model.CNNTokenTransformer
    RandomForestBinaryClassifier = _parent_model.RandomForestBinaryClassifier
    OneClassSVMClassifier = _parent_model.OneClassSVMClassifier
    PURandomForestClassifier = _parent_model.PURandomForestClassifier
    TwoStepPULearning = _parent_model.TwoStepPULearning


class CNN(nn.Module):
    """Plain CNN used by legacy training scripts (3 conv blocks + FC layers)."""

    def __init__(self, prior, input_dim, input_shape=None):
        super().__init__()
        self.prior = prior

        if input_shape is not None:
            self.channels, self.img_height, self.img_width = input_shape
        else:
            # Fall back to square inputs with 12 channels (old convention)
            self.channels = 12
            side = int((input_dim / self.channels) ** 0.5)
            self.img_height = self.img_width = side

        print(f"CNN 初始化: 输入维度 {input_dim}, 图像尺寸 {self.img_height}x{self.img_width}, 通道数 {self.channels}")

        # Convolutional backbone
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(128)

        # Determine pooling strategy (max 3 times while keeping at least 4x4 map)
        feature_height = self.img_height
        feature_width = self.img_width
        num_pools = 0
        while min(feature_height, feature_width) >= 4 and num_pools < 3:
            feature_height //= 2
            feature_width //= 2
            num_pools += 1

        self.num_pools = num_pools
        final_height = max(1, feature_height)
        final_width = max(1, feature_width)
        print(f"CNN 将进行 {num_pools} 次池化，最终特征图尺寸: {final_height}x{final_width}")

        fc_input_dim = 128 * final_height * final_width
        self.fc1 = nn.Linear(fc_input_dim, 300)
        self.bn_fc1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 100)
        self.bn_fc2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.bn_conv1(self.conv1(x)))
        if self.num_pools >= 1:
            h = F.max_pool2d(h, 2)

        h = F.relu(self.bn_conv2(self.conv2(h)))
        if self.num_pools >= 2:
            h = F.max_pool2d(h, 2)

        h = F.relu(self.bn_conv3(self.conv3(h)))
        if self.num_pools >= 3:
            h = F.max_pool2d(h, 2)

        h = h.reshape(h.size(0), -1)
        h = F.relu(self.bn_fc1(self.fc1(h)))
        h = F.relu(self.bn_fc2(self.fc2(h)))
        h = self.fc3(h)
        return h


class CNNTransformer(nn.Module):
    @staticmethod
    def from_pretrained(prior, input_dim, feature_map_size=None, cnn_features=None, projection_dim=None):
        """
        从预训练权重创建一个兼容的CNNTransformer模型
        
        Args:
            prior: 先验概率
            input_dim: 输入维度
            feature_map_size: 特征图大小
            cnn_features: CNN特征数
            projection_dim: 投影层输出维度
        
        Returns:
            兼容的CNNTransformer模型
        """
        # 创建模型的基本参数
        model = CNNTransformer(prior, input_dim)
        
        # 如果提供了额外参数，则调整模型结构
        if feature_map_size is not None and cnn_features is not None:
            # 计算投影层输入维度
            projection_input_dim = cnn_features * feature_map_size * feature_map_size
            
            # 重新创建投影层
            if projection_dim is not None:
                model.projection = nn.Linear(projection_input_dim, projection_dim)
        
        return model

    @staticmethod
    def create_compatible_model(prior, saved_state):
        """
        根据保存的状态创建兼容的模型
        
        Args:
            prior: 先验概率
            saved_state: 保存的模型状态字典
        
        Returns:
            兼容的CNNTransformer模型
        """
        # 检查是否有全部必要的层信息
        if "conv1.weight" in saved_state and "projection.weight" in saved_state:
            # 获取CNN参数
            conv1_in_channels = saved_state["conv1.weight"].shape[1]  # 应该是11
            conv1_out_channels = saved_state["conv1.weight"].shape[0]
            conv2_out_channels = saved_state["conv2.weight"].shape[0]
            conv3_out_channels = saved_state["conv3.weight"].shape[0]
            
            # 获取投影层参数
            projection_shape = saved_state["projection.weight"].shape
            projection_input_dim = projection_shape[1]  # 2048
            projection_output_dim = projection_shape[0]  # 128
            
            # 计算特征图大小: projection_input_dim = conv3_out_channels * feature_size^2
            feature_size = int((projection_input_dim / conv3_out_channels) ** 0.5)  # 4
            
            # 计算原始输入图像大小 (假设经过了3次池化，每次缩小一半)
            img_size = feature_size * 8  # 2^3 = 8 (假设三次池化)
            input_dim = conv1_in_channels * img_size * img_size
            
            print(f"创建兼容模型:")
            print(f"输入通道数: {conv1_in_channels}")
            print(f"特征图大小: {feature_size}x{feature_size}")
            print(f"原始图像大小: {img_size}x{img_size}")
            print(f"投影层输入维度: {projection_input_dim}")
            print(f"投影层输出维度: {projection_output_dim}")
            
            # 获取FC层参数
            fc1_out_features = saved_state["fc1.weight"].shape[0]
            fc2_out_features = saved_state["fc2.weight"].shape[0]
            
            # 创建完整兼容模型
            model = CNNTransformer(
                prior=prior, 
                input_dim=input_dim,
                conv_channels=[conv1_out_channels, conv2_out_channels, conv3_out_channels],
                projection_input_dim=projection_input_dim,
                projection_output_dim=projection_output_dim,
                fc_features=[fc1_out_features, fc2_out_features],
                input_channels=conv1_in_channels
            )
            
            return model
        else:
            raise ValueError("保存的状态中缺少必要的层信息")

    def __init__(self, prior, input_dim, conv_channels=None, projection_input_dim=None, 
                 projection_output_dim=128, fc_features=None, input_channels=11):
        super(CNNTransformer, self).__init__()
        self.prior = prior
        self.input_dim = input_dim
        self.input_channels = input_channels
        
        # 计算输入图像尺寸 (假设输入图像是正方形的)
        self.img_size = int((input_dim / self.input_channels) ** 0.5)
        
        # 设置默认卷积通道数
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        
        # 设置默认全连接层特征数
        if fc_features is None:
            fc_features = [300, 100]
        
        # 卷积层
        self.conv1 = nn.Conv2d(self.input_channels, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channels[2])
        
        # 计算特征图大小 (经过3次池化后)
        feature_map_size = self.img_size // 8
        if feature_map_size < 1:
            feature_map_size = 1
            print(f"警告: 输入图像过小，池化后特征图大小为1")
        
        # 计算展平后的特征维度
        if projection_input_dim is None:
            projection_input_dim = conv_channels[2] * feature_map_size * feature_map_size
        
        # 投影层
        self.projection = nn.Linear(projection_input_dim, projection_output_dim)
        
        # Transformer部分
        self.transformer = TransformerBlock(projection_output_dim)
        
        # 分类器部分
        self.fc1 = nn.Linear(projection_output_dim, fc_features[0])
        self.bn_fc1 = nn.BatchNorm1d(fc_features[0])
        self.fc2 = nn.Linear(fc_features[0], fc_features[1])
        self.bn_fc2 = nn.BatchNorm1d(fc_features[1])
        self.fc3 = nn.Linear(fc_features[1], 1)
        
        print(f"CNNTransformer初始化完成:")
        print(f"输入维度: {input_dim}, 图像尺寸: {self.img_size}x{self.img_size}")
        print(f"将进行 3 次池化，最终特征图尺寸: {feature_map_size}x{feature_map_size}")

    def forward(self, x):
        # 确保输入形状正确
        batch_size = x.size(0)
        if x.dim() == 2:
            # 如果输入是展平的，调整形状为 [batch_size, channels, height, width]
            x = x.view(batch_size, self.input_channels, self.img_size, self.img_size)
        
        # CNN部分
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # 展平特征
        x = x.view(batch_size, -1)
        
        # 投影到Transformer输入空间
        x = self.projection(x)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 分类器
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        
        return x.view(-1)
    
    def error(self, x, t):
        """计算分类错误率"""
        y = self.forward(x)
        pred = torch.sign(y)
        return (pred != t).float().mean().item()


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        # 自注意力部分 (处理批次优先格式)
        # x 形状: [batch_size, dim]
        x_norm = self.norm1(x)
        
        # 添加序列维度，调整为注意力层需要的形状
        x_norm = x_norm.unsqueeze(0)  # [1, batch_size, dim]
        
        # 应用自注意力
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        
        # 移除序列维度
        attn_output = attn_output.squeeze(0)  # [batch_size, dim]
        
        # 残差连接
        x = x + attn_output
        
        # MLP部分
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        
        # 残差连接
        x = x + mlp_output
        
        return x
