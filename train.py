"""训练脚本：对标 numpy 版的「多轮 + 验证集早停 + 回滚最佳权重 + 测试集评估」，
但换成 mini-batch + Adam，并且把最佳权重存成 state_dict。

用法：
    python train.py --data "C:\\Users\\admin\\Downloads\\croatianFishDataset\\croatianFishDataset1-0Dir"
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FishDataset
from model import build_model


def class_weights(dataset, num_classes):
    """和 numpy 版一致：w[c] = N / (K * count[c])，用来补偿类别不平衡。"""
    labels = np.array([y for _, y in dataset.items])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, criterion, num_classes, device):
    """返回 (loss, acc, macroF1, 混淆矩阵)。推理时不需要 softmax，argmax 直接用在 logits 上。"""
    model.eval()
    total_loss, total = 0.0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        total += y.size(0)
        for t, p in zip(y.tolist(), logits.argmax(dim=1).tolist()):
            cm[t, p] += 1
    acc = np.trace(cm) / max(cm.sum(), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return total_loss / max(total, 1), float(acc), float(f1.mean()), cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="croatianFishDataset1-0Dir 目录")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--filters", type=int, default=16)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--split", choices=["random", "ordered"], default="random",
                   help="ordered 复刻原版按文件名顺序切分")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0, help="Windows 下建议 0")
    p.add_argument("--out", default="runs/fish")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = FishDataset(args.data, "train", mode=args.split, seed=args.seed)
    val_ds = FishDataset(args.data, "val", mode=args.split, seed=args.seed)
    test_ds = FishDataset(args.data, "test", mode=args.split, seed=args.seed)
    num_classes = int(max(y for _, y in train_ds.items)) + 1
    print(f"split={args.split}  train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}  classes={num_classes}  device={device}")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=num_classes, filters=args.filters).to(device)
    weight = class_weights(train_ds, num_classes).to(device)
    print("class weights:", np.round(weight.cpu().numpy(), 2))
    # 与 numpy 版 -w[y]*log(p[y]) 同构；mean 会再除以 batch 内权重和，只差一个常数倍
    criterion = nn.CrossEntropyLoss(weight=weight)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc, bad_epochs, best_state, best_epoch = -1.0, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, seen = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)   # 千万别漏，否则梯度会累加
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            seen += y.size(0)

        v_loss, v_acc, v_f1, _ = evaluate(model, val_ld, criterion, num_classes, device)
        print(f"epoch {epoch:2d}  train_loss={run_loss / seen:.4f}  val_loss={v_loss:.4f}  "
              f"val_acc={v_acc:.4f}  macroF1={v_f1:.4f}  {time.time() - t0:.1f}s")

        if v_acc > best_acc:
            best_acc, bad_epochs, best_epoch = v_acc, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"验证集连续 {args.patience} 轮不升，早停于 epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n回滚到 val_acc 最好的 epoch {best_epoch}（{best_acc:.4f}），在测试集上评估：")
    t_loss, t_acc, t_f1, cm = evaluate(model, test_ld, criterion, num_classes, device)
    print(f"test  loss={t_loss:.4f}  acc={t_acc:.4f}  macroF1={t_f1:.4f}")
    print("混淆矩阵（行=真实，列=预测）：")
    print(cm)


if __name__ == "__main__":
    main()