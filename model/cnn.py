import torch.nn as nn
import torch.nn.functional as F
from .base import MyClassifier
import torch

class CNN(MyClassifier):
    def __init__(self, prior, input_dim, input_shape=None):
        super(CNN, self).__init__(prior)
        self.prior = prior

        # 解析输入维度
        if input_shape is not None:
            self.channels, self.img_height, self.img_width = input_shape
        else:
            # 回退到旧逻辑：假设平方图像
            self.channels = 12
            side = int((input_dim / self.channels) ** 0.5)
            self.img_height = self.img_width = side

        print(f"检测到输入图像尺寸: {self.img_height}x{self.img_width}")

        # 卷积层
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(128)
        
        # 计算经过池化后的特征图尺寸
        # 每次池化后尺寸减半
        feature_height = self.img_height
        feature_width = self.img_width
        num_pools = 0

        # 计算可以进行多少次池化（确保最小尺寸至少为2）
        while min(feature_height, feature_width) >= 4:
            feature_height //= 2
            feature_width //= 2
            num_pools += 1
            if num_pools >= 3:  # 最多进行3次池化
                break

        # 最终特征图尺寸
        final_height = max(1, feature_height)
        final_width = max(1, feature_width)
        self.num_pools = num_pools

        print(f"将进行 {num_pools} 次池化，最终特征图尺寸: {final_height}x{final_width}")

        # 计算全连接层的输入维度
        fc_input_dim = 128 * final_height * final_width
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input_dim, 300)
        self.bn_fc1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 100)
        self.bn_fc2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 1)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 卷积层
        h = F.relu(self.bn_conv1(self.conv1(x)))
        if self.num_pools >= 1:
            h = F.max_pool2d(h, 2)
        
        h = F.relu(self.bn_conv2(self.conv2(h)))
        if self.num_pools >= 2:
            h = F.max_pool2d(h, 2)
        
        h = F.relu(self.bn_conv3(self.conv3(h)))
        if self.num_pools >= 3:
            h = F.max_pool2d(h, 2)
        
        # 展平
        h = h.reshape(h.size(0), -1)
        
        # 全连接层
        h = F.relu(self.bn_fc1(self.fc1(h)))
        h = F.relu(self.bn_fc2(self.fc2(h)))
        h = self.fc3(h)
        return h

    def compute_loss(self, x, t, loss_func):
        self.loss = None
        h = self.forward(x)
        # 添加L2正则化，但使用较小的系数
        l2_lambda = 0.0001  # 减小正则化强度
        l2_reg = torch.tensor(0.).to(x.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        self.loss = loss_func(h.view(-1), t) + l2_lambda * l2_reg
        return self.loss 