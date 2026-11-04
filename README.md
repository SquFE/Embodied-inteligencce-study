# Embodied Intelligence Study

深度学习与具身智能学习记录。

## Week 2：Fashion-MNIST MLP

核心代码位于 `week2/`，包含模型、训练循环、评估和实验记录工具。

```bash
cd week2
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_mlp.py --train-size 512 --val-size 256 --epochs 2
python train_mlp.py --lr 0.1 --epochs 5
```

首次运行会下载 Fashion-MNIST；已有数据时使用本地缓存。默认使用 12000 张训练图、2000 张验证图、128 个隐藏单元、SGD、batch size 128 和随机种子 42。输出为 10 类 logits，使用交叉熵训练。

- `train_mlp.py`：独立训练与评估入口。
- `student_models.py` / `student_training.py`：Notebook 学习使用的模型及训练模块。
- `study_utils.py`：数据划分、加载、训练调度、绘图和结果保存工具。

实验结果写入 `week2/runs/`。训练配置选定后，可加 `--evaluate-test` 做最终测试集评估，不使用测试集调参。

## 验证记录

2026-11-04：512 张训练图、256 张验证图、2 轮的小规模流程检查通过，验证准确率 49.22%；该结果仅用于检查训练流程，不作为正式模型性能结论。

学习材料参考《动手学深度学习》，辅助工具由 AI 协助准备，核心模型、训练与评估作为学习练习完成。
