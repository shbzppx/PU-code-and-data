import torch.nn as nn
import torch.nn.functional as F
from .base import MyClassifier
import torch

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio), bias=False),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim, bias=False)
        )

    def forward(self, x):
        # x shape: (seq_len, batch_size, dim)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class CNNTransformer(MyClassifier):
    def __init__(self, prior, input_dim, input_shape=None):
        super(CNNTransformer, self).__init__(prior)
        self.prior = prior

        # 解析输入维度
        if input_shape is not None:
            self.channels, self.img_height, self.img_width = input_shape
        else:
            self.channels = 12
            side = int((input_dim / self.channels) ** 0.5)
            self.img_height = self.img_width = side

        print(f"检测到输入图像尺寸: {self.img_height}x{self.img_width}")

        # 卷积层
        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
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

        # 计算CNN输出的特征维度
        self.feature_dim = 128 * final_height * final_width
        
        # Transformer 参数
        self.transformer_dim = 128  # Transformer 特征维度
        self.num_heads = 4  # 注意力头数
        self.seq_length = final_height * final_width  # 序列长度
        
        # CNN特征到Transformer的投影层
        self.orig_feature_dim = self.feature_dim
        self.projection = nn.Linear(self.feature_dim, self.transformer_dim)
        
        # Transformer 层
        self.transformer = TransformerBlock(
            dim=self.transformer_dim,
            num_heads=self.num_heads,
            mlp_ratio=4.0,
            dropout=0.1
        )
        
        # 全连接层
        self.fc1 = nn.Linear(self.transformer_dim, 300)
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
        print(f'推理/当前模型 self.feature_dim =', self.feature_dim)

    def forward(self, x):
        # CNN特征提取
        h = F.relu(self.bn1(self.conv1(x)))
        if self.num_pools >= 1:
            h = F.max_pool2d(h, 2)
        
        h = F.relu(self.bn2(self.conv2(h)))
        if self.num_pools >= 2:
            h = F.max_pool2d(h, 2)
        
        h = F.relu(self.bn3(self.conv3(h)))
        if self.num_pools >= 3:
            h = F.max_pool2d(h, 2)
        
        # 展平操作以获取CNN特征
        batch_size = h.size(0)
        h_flat = h.view(batch_size, -1)
        
        # 投影到Transformer维度
        h_proj = self.projection(h_flat)
        
        # 重塑为序列格式 (seq_len, batch_size, dim)
        h_seq = h_proj.view(1, batch_size, self.transformer_dim)
        
        # 应用Transformer
        h_trans = self.transformer(h_seq)
        
        # 由于我们只有一个序列位置，取第一个输出
        h_trans = h_trans.view(batch_size, self.transformer_dim)
        
        # 全连接层
        h = F.relu(self.bn_fc1(self.fc1(h_trans)))
        h = F.relu(self.bn_fc2(self.fc2(h)))
        h = self.fc3(h)
        return h

    def compute_loss(self, x, t, loss_func):
        self.loss = None
        h = self.forward(x)
        l2_lambda = 0.0001
        l2_reg = torch.tensor(0.).to(x.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        self.loss = loss_func(h.view(-1), t) + l2_lambda * l2_reg
        return self.loss

    def load_my_state_dict(self, state_dict):
        """
        自定义加载状态字典的方法，处理输入尺寸不匹配的情况
        """
        own_state = self.state_dict()
        
        # 检查投影层权重是否尺寸不匹配
        if 'projection.weight' in state_dict:
            saved_proj_shape = state_dict['projection.weight'].shape
            current_proj_shape = own_state['projection.weight'].shape
            
            if saved_proj_shape != current_proj_shape:
                print(f"投影层尺寸不匹配: 保存的 {saved_proj_shape} vs 当前 {current_proj_shape}")
                print("重新初始化投影层和全连接层...")
                
                # 只加载匹配的层
                for name, param in state_dict.items():
                    if name.startswith('conv') or name.startswith('bn'):
                        if name in own_state and own_state[name].shape == param.shape:
                            own_state[name].copy_(param)
                
                return
        
        # 正常加载所有匹配的参数
        for name, param in state_dict.items():
            if name in own_state:
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                else:
                    print(f"参数 {name} 尺寸不匹配: {param.shape} vs {own_state[name].shape}")
            else:
                print(f"未使用参数 {name}") 