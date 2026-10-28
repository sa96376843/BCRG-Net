"""Additional compact segmentation comparators with no pretrained weights."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvBlock


def up_to(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)


class UNetPlusPlus(nn.Module):
    """Four-resolution UNet++ with nested, convolutional skip pathways."""
    def __init__(self, base=8):
        super().__init__()
        self.graph = None
        self.e0 = ConvBlock(3, base)
        self.e1 = ConvBlock(base, base * 2)
        self.e2 = ConvBlock(base * 2, base * 4)
        self.e3 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.c01 = ConvBlock(base + base * 2, base)
        self.c11 = ConvBlock(base * 2 + base * 4, base * 2)
        self.c21 = ConvBlock(base * 4 + base * 8, base * 4)
        self.c02 = ConvBlock(base * 4, base)
        self.c12 = ConvBlock(base * 8, base * 2)
        self.c03 = ConvBlock(base * 5, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x, mask=None):
        x00 = self.e0(x)
        x10 = self.e1(self.pool(x00))
        x20 = self.e2(self.pool(x10))
        x30 = self.e3(self.pool(x20))
        x01 = self.c01(torch.cat([x00, up_to(x10, x00)], 1))
        x11 = self.c11(torch.cat([x10, up_to(x20, x10)], 1))
        x02 = self.c02(torch.cat([x00, x01, up_to(x11, x00)], 1))
        x21 = self.c21(torch.cat([x20, up_to(x30, x20)], 1))
        x12 = self.c12(torch.cat([x10, x11, up_to(x21, x10)], 1))
        x03 = self.c03(torch.cat([x00, x01, x02, up_to(x12, x00)], 1))
        return self.head(x03), {}


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, dilation=1, groups=1):
        padding = dilation if kernel == 3 else 0
        super().__init__(
            nn.Conv2d(cin, cout, kernel, padding=padding, dilation=dilation,
                      groups=groups, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SeparableConv(nn.Sequential):
    def __init__(self, cin, cout, dilation=1):
        super().__init__(
            ConvBNReLU(cin, cin, 3, dilation, groups=cin),
            ConvBNReLU(cin, cout, 1))


class CompactDeepLabV3Plus(nn.Module):
    """DeepLabV3+ comparator with a compact non-pretrained encoder and ASPP."""
    def __init__(self, base=8):
        super().__init__()
        self.graph = None
        self.e1 = ConvBlock(3, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        c = base * 8
        hidden = base * 2
        self.aspp = nn.ModuleList([
            ConvBNReLU(c, hidden, 1),
            SeparableConv(c, hidden, 2),
            SeparableConv(c, hidden, 4),
            SeparableConv(c, hidden, 8),
        ])
        self.global_projection = ConvBNReLU(c, hidden, 1)
        self.project = ConvBNReLU(hidden * 5, c, 1)
        self.low = ConvBNReLU(base, base * 2, 1)
        self.decoder = nn.Sequential(
            SeparableConv(c + base * 2, base * 4),
            SeparableConv(base * 4, base * 4),
            nn.Conv2d(base * 4, 1, 1))

    def forward(self, x, mask=None):
        low = self.e1(x)
        z = self.e2(self.pool(low))
        z = self.e3(self.pool(z))
        z = self.e4(self.pool(z))
        parts = [branch(z) for branch in self.aspp]
        pooled = F.adaptive_avg_pool2d(z, 1)
        pooled = up_to(self.global_projection(pooled), z)
        z = self.project(torch.cat(parts + [pooled], 1))
        low = self.low(low)
        z = torch.cat([up_to(z, low), low], 1)
        return self.decoder(z), {}


def build_additional_baseline(method, base=8):
    if method == 'unetpp':
        return UNetPlusPlus(base)
    if method == 'deeplab':
        return CompactDeepLabV3Plus(base)
    return None
