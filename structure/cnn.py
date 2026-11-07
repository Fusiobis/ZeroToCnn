from pathlib import Path
from PIL import Image
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from relu import ReLU
def load_croatian(root, size=32, val_ratio=0.15, test_ratio=0.15):
    root = Path(root)
    dirs = [p for p in root.iterdir() if p.is_dir()]          # 别把 metadata.xml 也 int()
    train, val, test = [], [], []
    for d in sorted(dirs, key=lambda p: int(p.name)):         # 按数字排序，不是字符串！
        label = int(d.name)
        files = sorted(d.glob("*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        n = len(files)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        for k, f in enumerate(files):
            if k >= n - n_test:                               # 类内末尾做 test
                test.append((f, label))
            elif k >= n - n_test - n_val:
                val.append((f, label))
            else:
                train.append((f, label))

    def to_array(path):
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        # 保持长宽比补成正方形（边缘像素补齐，不补黑边）
        arr = np.pad(arr, ((top, side - h - top), (left, side - w - left), (0, 0)), mode="edge")
        im = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)               # (size, size, 3)，保持 0~255

    def pack(items):
        return (np.stack([to_array(p) for p, _ in items]),
                np.array([c for _, c in items], dtype=np.int64))

    return (*pack(train), *pack(val), *pack(test))
'''
conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(15 * 15 * 8, 12)
'''

IMG = 32
NUM_CLASSES = 12
POOL = (IMG - 2) // 2                        # 32 → 15

DATA_ROOT = Path(r"C:\Users\admin\Downloads\croatianFishDataset\croatianFishDataset1-0Dir")

# ---- 补 1：原文件定义了 load_croatian 却从未调用，xtr/ytr 全程未定义 ----
xtr, ytr, xva, yva, xte, yte = load_croatian(DATA_ROOT, size=IMG)
print(f"train={len(xtr)}  val={len(xva)}  test={len(xte)}  img={IMG}")

conv = Conv3x3(16)                           # 8 太少，16 起步
pool = MaxPool2()
relu = ReLU()
softmax = Softmax(POOL * POOL * 16, NUM_CLASSES)   # 15*15*16 = 3600

weights = np.array([len(ytr) / (NUM_CLASSES * c) for c in np.bincount(ytr, minlength=NUM_CLASSES)])
print("class weights:", np.round(weights, 2))

def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = relu.forward(out)
    out = pool.forward(out)
    out = softmax.forward(out)
    
    #loss = -np.log(out[label])
    loss = -weights[label] * np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc
def train(im, label, lr=.005):
    out, loss, acc = forward(im, label)

    #gradient = np.zeros(10)
    #gradient[label] = -1 / out[label]
    gradient = np.zeros(NUM_CLASSES)
    gradient[label] = -weights[label] / out[label]

    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = relu.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc
print("CNN initialized!")

def evaluate(xs, ys):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    total_loss = 0.0
    for im, y in zip(xs, ys):
        out, l, _ = forward(im, y)
        total_loss += l
        cm[y, int(np.argmax(out))] += 1

    acc = np.trace(cm) / max(cm.sum(), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return total_loss / len(ys), acc, float(f1.mean()), cm
'''
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

  # Train!
  loss = 0
  num_correct = 0
  for i, (im, label) in enumerate(zip(train_images, train_labels)):
    if i > 0 and i % 100 == 99:
      print(
        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
        (i + 1, loss / 100, num_correct)
      )
      loss = 0
      num_correct = 0

    l, acc = train(im, label)
    loss += l
    num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
'''
best_acc, bad, best = -1.0, 0, None

for epoch in range(1, 11):
    print(f'--- Epoch {epoch} ---')
    permutation = np.random.permutation(len(xtr))
    xtr, ytr = xtr[permutation], ytr[permutation]

    loss_sum = 0.0
    for i, (im, label) in enumerate(zip(xtr, ytr)):
        l, _ = train(im, label, lr=0.01)
        loss_sum += l
        if i % 100 == 99:
            print(f'[Step {i+1}] loss={loss_sum/100:.3f}')
            loss_sum = 0.0

    v_loss, v_acc, v_f1, _ = evaluate(xva, yva)
    print(f'val  loss={v_loss:.4f}  acc={v_acc:.4f}  macroF1={v_f1:.4f}')

    if v_acc > best_acc:
        best_acc, bad = v_acc, 0
        best = (conv.filters.copy(), softmax.weights.copy(), softmax.biases.copy())
    else:
        bad += 1
        if bad >= 3:
            print(f'val 连续 3 轮不升，早停于 epoch {epoch}')
            break

# 训练结束后恢复验证集最好的那一轮，再用它跑测试集
if best is not None:
    conv.filters[...] = best[0]
    softmax.weights[...] = best[1]
    softmax.biases[...] = best[2]