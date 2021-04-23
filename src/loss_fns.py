import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


def focal_loss(input_values, gamma):
    """Based on https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    """Based on https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""

    def __init__(self, weight: Optional[torch.Tensor], gamma: float):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(
            F.cross_entropy(input, target, reduction="none", weight=self.weight),
            self.gamma,
        )


class LDAMLoss(nn.Module):
    """Based on https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""

    def __init__(
        self,
        weight: Optional[torch.Tensor],
        num_per_class: List[int],
        max_m: float,
        s: float,
    ):
        super(LDAMLoss, self).__init__()
        cls_num_list = np.array(num_per_class)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.as_tensor(m_list, dtype=torch.get_default_dtype())
        self.register_buffer("m_list", m_list)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        # TODO: review code
        # TODO: what's the expected shape?
        index = torch.zeros_like(x, dtype=torch.long)
        index.scatter_(1, target.view(-1, 1), 1)

        batch_m = torch.matmul(self.m_list[None, :], index.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
