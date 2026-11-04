import torch
from torch import nn


class MLP(nn.Module):
    """28x28 灰度图 -> 隐藏层 -> ReLU -> 可选 Dropout -> 10 类 logits。"""
    def __init__(self, hidden=128, dropout=0.):
        super().__init__()
        # TODO 构建 self.net；Flatten 保留 batch 维，最后不要添加 Softmax。
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 10))

    def forward(self, images):
        # TODO 返回 self.net(images)。
        return self.net(images)
