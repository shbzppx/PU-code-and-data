import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import MyClassifier


class CNNTokenTransformer(MyClassifier):
    """CNN feature-token Transformer with scalar score output.

    The model keeps the spatial feature map after the second convolution,
    treats each spatial location as one token, and applies self-attention
    across those tokens. The final scalar score is compatible with the
    existing PU loss pipeline.
    """

    def __init__(
        self,
        prior,
        input_dim,
        input_shape=None,
        transformer_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super(CNNTokenTransformer, self).__init__(prior)
        self.prior = prior

        if input_shape is not None:
            self.channels, self.img_height, self.img_width = input_shape
        else:
            self.channels = 12
            side = int((input_dim / self.channels) ** 0.5)
            self.img_height = self.img_width = side

        self.transformer_dim = int(transformer_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)

        print(f"检测到输入图像尺寸: {self.img_height}x{self.img_width}")

        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        token_height = max(1, int(self.img_height) // 2)
        token_width = max(1, int(self.img_width) // 2)
        self.token_grid = (token_height, token_width)
        self.num_tokens = int(token_height * token_width)

        print(
            f"CNNTokenTransformer token网格: {token_height}x{token_width}, "
            f"token数: {self.num_tokens}"
        )

        self.token_projection = nn.Linear(64, self.transformer_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.transformer_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=self.num_heads,
            dim_feedforward=self.transformer_dim * 2,
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.transformer_dim)

        self.fc1 = nn.Linear(self.transformer_dim, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))

        batch_size, channels, height, width = h.shape
        tokens = h.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        tokens = self.token_projection(tokens)

        if tokens.shape[1] != self.pos_embed.shape[1]:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed

        tokens = tokens + pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        h = tokens.mean(dim=1)

        h = F.relu(self.bn_fc1(self.fc1(h)))
        h = self.dropout(h)
        h = F.relu(self.bn_fc2(self.fc2(h)))
        h = self.dropout(h)
        return self.fc3(h)

    def compute_loss(self, x, t, loss_func):
        self.loss = None
        h = self.forward(x)
        l2_lambda = 0.0001
        l2_reg = torch.tensor(0.0, device=x.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        self.loss = loss_func(h.view(-1), t) + l2_lambda * l2_reg
        return self.loss
