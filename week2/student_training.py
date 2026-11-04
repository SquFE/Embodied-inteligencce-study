import torch
from torch import nn


def train_one_epoch(model, loader, criterion, optimizer):
    """更新一轮；返回按样本数加权的平均训练损失。"""
    model.train()
    total_loss, count = 0., 0
    for images, labels in loader:
        # TODO 清零梯度，前向得到 logits，计算标量交叉熵；检查 logits/labels 的形状。
        optimizer.zero_grad()
        logits=model(images)
        assert logits.ndim == 2 and logits.shape[0] == len(labels)
        assert labels.ndim == 1 and labels.dtype == torch.long
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('训练 loss 出现 NaN/Inf；结果会由 fit 保存后再检查。')
        # TODO 反向传播并用优化器更新参数。
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    return total_loss / count


def evaluate(model, loader):
    """返回整个数据集的平均交叉熵与准确率；不更新参数。"""
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for images, labels in loader:
            # TODO 前向、累加 loss、计算类别预测和正确数量；不能简单平均各 batch 的平均值。
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
            predictied = logits.argmax(dim=1)
            correct += (predictied == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total
