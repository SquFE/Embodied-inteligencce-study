"""教学辅助：只封装数据下载、记录和绘图；关键训练步骤在练习中实现。"""
from pathlib import Path
from functools import lru_cache
import csv
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}
BASE_URL = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def configure_cpu(threads=2):
    """小模型使用有限线程，避免过多并行开销；不是对所有项目都最优。"""
    torch.set_num_threads(threads)

def prepare_data(include_test=False):
    directory = ROOT / 'data' / 'fashion-mnist'
    directory.mkdir(parents=True, exist_ok=True)
    selected = FILES if include_test else {k:v for k,v in FILES.items() if k.startswith('train-')}
    for name, digest in selected.items():
        target = directory / name
        if target.exists() and hashlib.md5(target.read_bytes()).hexdigest() == digest:
            continue
        temporary = target.with_suffix('.gz.part')
        error = None
        for attempt in range(3):
            try:
                print(f'Downloading {name}, attempt {attempt+1}/3 ...', flush=True)
                with urllib.request.urlopen(BASE_URL + name, timeout=30) as response, temporary.open('wb') as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                if hashlib.md5(temporary.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'MD5 mismatch: {name}')
                temporary.replace(target)
                error = None
                break
            except Exception as exc:
                error = exc
                temporary.unlink(missing_ok=True)
        if error is not None:
            raise RuntimeError(f'Download failed for {name}; rerun prepare_data later. {error}') from error
    return directory

@lru_cache(maxsize=2)
def _raw_data(split):
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    directory = prepare_data(include_test=split == 'test')
    prefix = 'train' if split == 'train' else 't10k'
    with gzip.open(directory / f'{prefix}-images-idx3-ubyte.gz', 'rb') as stream:
        raw = stream.read()
    magic, count, rows, columns = struct.unpack('>IIII', raw[:16])
    assert magic == 2051 and (rows, columns) == (28, 28)
    assert len(raw) == 16 + count * rows * columns
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 28, 28)
    with gzip.open(directory / f'{prefix}-labels-idx1-ubyte.gz', 'rb') as stream:
        raw_labels = stream.read()
    label_magic, label_count = struct.unpack('>II', raw_labels[:8])
    assert label_magic == 2049 and label_count == count and len(raw_labels) == 8 + count
    labels = np.frombuffer(raw_labels, dtype=np.uint8, offset=8)
    return images, labels

def _tensor_dataset(images, labels, indices):
    # 图片转 float32 并缩放到 [0,1]；类别编号转 long，shape 为 (N,)。
    x = torch.from_numpy(images[indices].copy()).unsqueeze(1).float().div_(255)
    y = torch.from_numpy(labels[indices].copy()).long()
    return TensorDataset(x, y)

def load_splits(train_size=12000, val_size=2000, seed=42):
    """从官方训练集固定划出 50000/10000 池，再取 CPU 学习用子集。不使用官方测试集。"""
    if not (1 <= train_size <= 50000 and 1 <= val_size <= 10000):
        raise ValueError('train_size in [1,50000], val_size in [1,10000]')
    images, labels = _raw_data('train')
    order = np.random.default_rng(seed).permutation(len(images))
    train_ids = order[:50000][:train_size]
    val_ids = order[50000:][:val_size]
    train = _tensor_dataset(images, labels, train_ids)
    val = _tensor_dataset(images, labels, val_ids)
    meta = dict(seed=seed, train_size=train_size, val_size=val_size,
                official_training_count=60000, official_test_used=False,
                split='seeded permutation; first 50000 training pool; last 10000 validation pool',
                train_indices_sha256=hashlib.sha256(train_ids.tobytes()).hexdigest(),
                val_indices_sha256=hashlib.sha256(val_ids.tobytes()).hexdigest(),
                transform='uint8 / 255 -> float32; shape N,1,28,28')
    return train, val, meta

def load_test():
    images, labels = _raw_data('test')
    return _tensor_dataset(images, labels, np.arange(len(images)))

def make_loaders(train, val, batch_size=128, seed=42):
    generator = torch.Generator().manual_seed(seed)
    # num_workers=0 适合 Windows/Notebook，不启动额外数据进程。
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    train_eval = DataLoader(train, batch_size=512, shuffle=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=512, shuffle=False, num_workers=0)
    return train_loader, train_eval, val_loader

def reference_evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, correct / total

def fit(model, train_loader, train_eval, val_loader, optimizer, epochs,
        train_one_epoch, evaluate):
    """训练步骤和评估函数由学习者传入。本函数负责轮次、计时和保留失败状态。"""
    history = []
    status = 'completed'
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs + 1):
        start = time.perf_counter()
        if epoch > 0:
            try:
                train_one_epoch(model, train_loader, criterion, optimizer)
            except FloatingPointError:
                status = 'non_finite_loss'
        train_loss, train_acc = evaluate(model, train_eval)
        val_loss, val_acc = evaluate(model, val_loader)
        row = dict(epoch=epoch, train_loss=float(train_loss), val_loss=float(val_loss),
                   train_acc=float(train_acc), val_acc=float(val_acc),
                   seconds=time.perf_counter() - start)
        history.append(row)
        if epoch in (0, 1, epochs) or epoch % max(1, epochs // 5) == 0:
            print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}", flush=True)
        if not all(math.isfinite(v) for v in (train_loss, val_loss)):
            status = 'non_finite_loss'
        if status != 'completed':
            break
    return history, status

def _git_info():
    git = shutil.which('git')
    bundled = Path(r'C:\Users\xisha\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe')
    if git is None and bundled.exists():
        git = str(bundled)
    if not git:
        return dict(commit=None, tracked_changes=None)
    try:
        commit = subprocess.check_output([git, '-C', str(ROOT), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        dirty = subprocess.check_output([git, '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], stderr=subprocess.DEVNULL, timeout=5).decode('utf-8', errors='replace')
        return dict(commit=commit, tracked_changes=dirty)
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, tracked_changes=None)

def _json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k:_json_safe(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

def save_run(name, model, history, config, status='completed', category='runs', source_file=None):
    """每次生成新目录，先保留结果再由调用方验收；不会覆盖前一次运行。"""
    import matplotlib
    # Notebook 已导入 pyplot 时保留 inline 后端；命令行首次绘图时使用 Agg。
    if 'matplotlib.pyplot' not in sys.modules:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('run name must use English letters, digits, - or _')
    if category not in ('runs', 'validation/reference_runs'):
        raise ValueError('unsupported output category')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / category / f'{stamp}_{name}'
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch','train_loss','val_loss','train_acc','val_acc','seconds'])
        writer.writeheader()
        writer.writerows(history)
    record = dict(config=config, status=status, final=history[-1], git=_git_info(),
                  python=sys.version, executable=sys.executable, torch=torch.__version__,
                  numpy=np.__version__, cpu_threads=torch.get_num_threads(), timezone='Asia/Shanghai')
    (out / 'config.json').write_text(json.dumps(_json_safe(record), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    torch.save(model.state_dict(), out / 'model.pt')
    if source_file is not None:
        source = Path(source_file)
        if source.is_file():
            shutil.copy2(source, out / source.name)
    shutil.copy2(__file__, out / 'study_utils_snapshot.py')
    module_names = ['training_reference.py','models_reference.py'] if category.startswith('validation/') else ['student_training.py','student_models.py']
    source_base = ROOT / 'solutions' if category.startswith('validation/') else ROOT
    for filename in module_names:
        source = source_base / filename
        if source.is_file():
            shutil.copy2(source, out / filename)
    record['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in out.iterdir() if p.suffix in ('.py','.ipynb')}
    (out / 'config.json').write_text(json.dumps(_json_safe(record), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, label in [('train', 'train'), ('val', 'validation')]:
        axes[0].plot([r['epoch'] for r in history], [r[key+'_loss'] for r in history], label=label)
        axes[1].plot([r['epoch'] for r in history], [r[key+'_acc'] for r in history], label=label)
    axes[0].set(xlabel='Epoch (0 = untrained)', ylabel='Cross entropy')
    axes[1].set(xlabel='Epoch (0 = untrained)', ylabel='Accuracy', ylim=(0,1))
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'curves.png', dpi=140)
    plt.close(fig)
    (out / 'notes.md').write_text('# 实验记录\n\n预期：\n\n实际结果：\n\n解释与失败分析：\n\n下一步：\n', encoding='utf-8')
    print('Saved:', out)
    return out

def read_runs(paths):
    records = []
    for path in paths:
        p = Path(path)
        record = json.loads((p/'config.json').read_text(encoding='utf-8'))
        records.append(dict(name=p.name, config=record['config'], status=record['status'], final=record['final']))
    return records

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args()
    if args.download:
        print(prepare_data(include_test=args.include_test))
