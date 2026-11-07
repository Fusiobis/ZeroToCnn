"""与 numpy 版 cnn.py / load_croatian 一致的划分与预处理，但返回 torch.Tensor。

预处理链路（和原版逐字一致）：
    RGB uint8 -> 边缘补齐成正方形 -> BILINEAR 缩放到 32x32 -> (H,W,3) float32
这里额外做两件事：
    1. 值域从 0~255 改成 0~1（后面模型里减 0.5，等价于原来的 image/255-0.5）
    2. HWC -> CHW，因为 PyTorch 的卷积要求通道在最前面
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def list_split(root, split="train", val_ratio=0.15, test_ratio=0.15,
               mode="random", seed=0):
    """返回 [(path, label), ...]。

    mode="ordered"：完全复刻原版 —— 每类按文件名末尾数字排序，末尾 test、往前 val、其余 train。
    mode="random" ：每类内部随机打乱后再切（推荐）。原版那种切法会让 test 全是"最后几张"，
                    和训练集分布不一致，实测测试集准确率会莫名低很多。
    """
    root = Path(root)
    dirs = [p for p in root.iterdir() if p.is_dir()]
    items = []
    for d in sorted(dirs, key=lambda p: int(p.name)):        # 按数字排序，不是字符串
        label = int(d.name)
        files = sorted(d.glob("*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        n = len(files)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))

        if mode == "ordered":
            lo, hi = {
                "train": (0, n - n_test - n_val),
                "val": (n - n_test - n_val, n - n_test),
                "test": (n - n_test, n),
            }[split]
            chosen = files[lo:hi]
        else:
            files = list(files)
            random.Random(seed * 1000 + label).shuffle(files)   # 每类独立打乱，可复现
            chosen = {
                "train": files[n_test + n_val:],
                "val": files[n_test:n_test + n_val],
                "test": files[:n_test],
            }[split]
        items += [(f, label) for f in chosen]
    return items


def to_array(path, size=32):
    """读图 -> 边缘补齐成正方形 -> 缩放 -> (H, W, 3) float32，值域 0~1"""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    # mode="edge" 复制边缘像素，不补黑边（和 numpy 版一致）
    arr = np.pad(arr, ((top, side - h - top), (left, side - w - left), (0, 0)), mode="edge")
    im = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


class FishDataset(Dataset):
    def __init__(self, root, split="train", size=32, val_ratio=0.15, test_ratio=0.15,
                 mode="random", seed=0):
        self.items = list_split(root, split, val_ratio, test_ratio, mode, seed)
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        hwc = to_array(path, self.size)                          # (32, 32, 3)
        chw = torch.from_numpy(hwc).permute(2, 0, 1).float()     # (3, 32, 32)
        return chw, label