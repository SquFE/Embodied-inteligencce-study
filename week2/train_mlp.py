"""最终验收：不看前面答案，完成模型、训练与评估。允许复用数据/记录工具。"""

from pathlib import Path
import argparse
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = next(
    p
    for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    if (p / "study_utils.py").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import study_utils as su

import torch
from torch import nn


def train_one_epoch(model, loader, criterion, optimizer):
    """更新一轮；返回按样本数加权的平均训练损失。"""
    model.train()
    total_loss, count = 0.0, 0
    for images, labels in loader:
        # TODO 清零梯度，前向得到 logits，计算标量交叉熵；检查 logits/labels 的形状。
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        assert logits.ndim == 2 and logits.shape[0] == len(labels)
        assert labels.ndim == 1 and labels.dtype == torch.long
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "训练 loss 出现 NaN/Inf；结果会由 fit 保存后再检查。"
            )
        # TODO 反向传播并用优化器更新参数。
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    return total_loss / count


def evaluate(model, loader):
    """返回整个数据集的平均交叉熵与准确率；不更新参数。"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for images, labels in loader:
            # TODO 前向、累加 loss、计算类别预测和正确数量；不能简单平均各 batch 的平均值。
            logits = model(images)
            loss = criterion(logits, labels).item()
            total += len(labels)
            loss_sum += loss
            predicted = logits.argmax(dim=1)
            correct += (labels == predicted).sum().item()
    return loss_sum / total, correct / total


def build_model(hidden=128, dropout=0.0):
    # TODO 独立写出 Flatten -> Linear -> ReLU -> Dropout -> Linear；输出10类 logits。
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 10),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Only after selecting config using validation data",
    )
    parser.add_argument("--reference-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, lr must be positive")
    su.configure_cpu(2)
    su.seed_everything(args.seed)
    train, val, split = su.load_splits(args.train_size, args.val_size, args.seed)
    train_loader, train_eval, val_loader = su.make_loaders(
        train, val, args.batch_size, args.seed
    )
    model = build_model(args.hidden, args.dropout)
    assert model(train.tensors[0][:2]).shape == (2, 10)
    # TODO 根据 args.optimizer 创建 SGD 或 Adam；设置 lr 与 weight_decay。
    optimizer_kind = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.Adam
    optimizer = optimizer_kind(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    config = vars(args).copy()
    config["split"] = split
    config["model"] = "mlp"
    history, status = su.fit(
        model,
        train_loader,
        train_eval,
        val_loader,
        optimizer,
        args.epochs,
        train_one_epoch,
        evaluate,
    )
    category = "validation/reference_runs" if args.reference_run else "runs"
    out = su.save_run(
        "final_mlp",
        model,
        history,
        config,
        status,
        category=category,
        source_file=Path(__file__),
    )
    if status != "completed":
        raise RuntimeError(f"Training failed; partial results saved in {out}")
    if args.evaluate_test:
        # 先用验证集决定配置，测试结果只用于最终报告，不再据此调参。
        test = su.load_test()
        test_loss, test_acc = evaluate(
            model, DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
        )
        import json

        (out / "test_evaluation.json").write_text(
            json.dumps(
                dict(
                    test_loss=test_loss,
                    test_accuracy=test_acc,
                    official_test_count=len(test),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Final test:", test_loss, test_acc)
    print("Finished. Validation accuracy:", history[-1]["val_acc"])
    print("Record your interpretation in:", out / "notes.md")


if __name__ == "__main__":
    main()
