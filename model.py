"""FishCNN：可开关的 深度可分离 / 注意力 / 激活 / 归一化。

兼容性：build_model(arch="plain", norm="none", act="relu", attn="none")
时，行为与原来的逐层翻译完全等价。
"""
import torch
import torch.nn as nn


# ------------------------------------------------------------------ 工厂函数
def auto_groups(ch):
    for g in (4, 8, 2):
        if ch % g == 0:
            return g
    return ch


def make_norm(kind, ch):
    if kind == "none":
        return nn.Identity()
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        g = auto_groups(ch)
        if ch % g:
            raise ValueError(f"GroupNorm: {g} 不能整除 {ch}")
        return nn.GroupNorm(g, ch)
    raise ValueError(f"未知 norm: {kind}")


def make_act(kind):
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "relu6":
        return nn.ReLU6(inplace=True)
    if kind == "gelu":
        return nn.GELU()
    if kind == "mish":
        return nn.Mish(inplace=True)
    raise ValueError(f"未知 act: {kind}")


def conv_out(h, k, s, p):
    return (h + 2 * p - k) // s + 1


# ------------------------------------------------------------------ 积木
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=0,
                 norm="bn", act="relu"):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride, padding,
                            groups=in_ch, bias=(norm == "none"))
        self.norm1 = make_norm(norm, in_ch)
        self.act1 = make_act(act)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=(norm == "none"))
        self.norm2 = make_norm(norm, out_ch)
        self.act2 = make_act(act)

    def forward(self, x):
        x = self.act1(self.norm1(self.dw(x)))
        return self.act2(self.norm2(self.pw(x)))


class SEBlock(nn.Module):
    def __init__(self, ch, r=2, act="relu"):
        super().__init__()
        hidden = max(1, ch // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, hidden, 1, bias=True)
        self.act = make_act(act)
        self.fc2 = nn.Conv2d(hidden, ch, 1, bias=True)

    def forward(self, x):
        s = self.pool(x)
        return x * torch.sigmoid(self.fc2(self.act(self.fc1(s))))


class CBAM(nn.Module):
    def __init__(self, ch, r=2, k=7, act="relu"):
        super().__init__()
        hidden = max(1, ch // r)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            make_act(act),
            nn.Conv2d(hidden, ch, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2, bias=True)

    def forward(self, x):
        ca = (self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
              + self.mlp(torch.amax(x, dim=(2, 3), keepdim=True)))
        x = x * torch.sigmoid(ca)
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(s))


# ------------------------------------------------------------------ 主模型
def _specs(arch, in_ch, filters):
    if arch == "plain":
        return [(in_ch, filters, 3, 1, 0)]
    if arch == "dw1":
        return [(in_ch, filters, 3, 1, 0)]
    if arch == "dw2":
        return [(in_ch, filters, 3, 1, 0), (filters, filters, 3, 1, 1)]
    raise ValueError(f"未知 arch: {arch}")


class FishCNN(nn.Module):
    def __init__(self, num_classes=12, in_ch=3, size=32, filters=16,
                 arch="dw2", act="relu", norm="bn", attn="se", attn_r=2):
        super().__init__()
        specs = _specs(arch, in_ch, filters)
        layers, h = [], size

        for ic, oc, k, s, p in specs:
            if arch == "plain":
                layers += [
                    nn.Conv2d(ic, oc, k, s, p, bias=(norm == "none")),
                    make_norm(norm, oc),
                    make_act(act),
                ]
            else:
                layers.append(DepthwiseSeparable(ic, oc, k, s, p, norm, act))
            h = conv_out(h, k, s, p)

        last_ch = specs[-1][1]
        if attn == "se":
            layers.append(SEBlock(last_ch, r=attn_r))
        elif attn == "cbam":
            layers.append(CBAM(last_ch, r=attn_r))
        elif attn != "none":
            raise ValueError(f"未知 attn: {attn}")

        layers.append(nn.MaxPool2d(2, 2))
        h //= 2

        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(last_ch * h * h, num_classes)

    def forward(self, x):
        x = x - 0.5                    # 与 numpy 版对齐，别删
        x = self.features(x)
        return self.fc(torch.flatten(x, 1))   # 返回 logits


def build_model(num_classes=12, filters=16, size=32, **kw):
    return FishCNN(num_classes=num_classes, filters=filters, size=size, **kw)