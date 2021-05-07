## Based on: https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/models/resnet18k.py
from .simple_models import Size
from .simclr.resnet_wider import ResNet as WiderResNet, Bottleneck, BasicBlock
from torchvision.models.resnet import model_urls
from torchvision.models.utils import load_state_dict_from_url
from typing import Optional

## ResNet18 for CIFAR
## Based on: https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py

import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class CIFARResNet(nn.Module):
    input_size = (3, 32, 32)

    def __init__(
        self, input_size: Size, output_size: Size, block, num_blocks, init_channels=64
    ):
        assert tuple(input_size) == self.input_size
        super(CIFARResNet, self).__init__()
        self.in_planes = init_channels
        c = init_channels

        self.conv1 = nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, c, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * c, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * c, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * c, num_blocks[3], stride=2)
        self.linear = nn.Linear(8 * c * block.expansion, output_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        # eg: [2, 1, 1, ..., 1]. Only the first one downsamples.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @property
    def linear_output(self):
        return self.linear


def make_resnet18k(input_size: Size, output_size: Size, k: int = 64) -> CIFARResNet:
    """ Returns a ResNet18 for CIFAR-sized data with width parameter k. (k=64 is standard ResNet18)"""
    return CIFARResNet(
        input_size, output_size, PreActBlock, [2, 2, 2, 2], init_channels=k
    )


class ResNet(WiderResNet):
    input_size = (3, 224, 224)

    def __init__(
        self,
        arch: str,
        input_size: Size,
        output_size: Size,
        pretrained: bool,
        ckpt_dir: Optional[str] = None,
    ):
        assert tuple(input_size) == self.input_size

        if arch == "resnet18":
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif arch == "resnet34":
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif arch == "resnet50":
            block = Bottleneck
            layers = [3, 4, 6, 3]
        else:
            raise ValueError(f"{arch} is not a supported type of ResNet!")

        super().__init__(block=block, layers=layers, width_mult=1)

        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls[arch], model_dir=ckpt_dir, progress=True
            )
            self.load_state_dict(state_dict)
