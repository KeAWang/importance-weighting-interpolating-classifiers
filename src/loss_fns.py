import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


def focal_loss(input_values, gamma):
    """Based on https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    """Based on https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""

    def __init__(self, weight: Optional[torch.Tensor], gamma: float, reduction: str):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        assert reduction == "none"
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, input, target):
        return focal_loss(
            F.cross_entropy(input, target, reduction="none", weight=self.weight),
            self.gamma,
        )


class LDAMLoss(nn.Module):
    """Reimplementation of https://github.com/kaidic/LDAM-DRW/blob/master/losses.py

    The original published code included a temperature hyperparameter, named `s` in the
    released code, that is by default set to 30. This was not mentioned in the paper.

    """

    def __init__(
        self,
        weight: Optional[torch.Tensor],
        num_per_class: List[int],
        max_margin: float,
        inv_temperature: float,
        reduction: str,
    ):
        super(LDAMLoss, self).__init__()
        assert max_margin > 0
        assert inv_temperature > 0
        # inverse temperature parameter not specified in paper
        # the released code sets this to 30 by default
        self.inv_temperature = inv_temperature

        if num_per_class is None:
            raise ValueError(
                f"num_per_class must be set to initialize {self.__class__}"
            )
        num_per_class = np.array(num_per_class)
        margins = num_per_class ** (-1 / 4)
        # Rescale to largest enforced margin
        margins = margins * (max_margin / np.max(margins))
        margins = torch.as_tensor(margins, dtype=torch.get_default_dtype())
        # margins is 1D Tensor of shape (k,) for k classes
        self.register_buffer("margins", margins)
        self.register_buffer("weight", weight)
        assert self.margins.shape == self.weight.shape

    def forward(self, logits, target):
        # follows the interface of torch.nn.CrossEntropyLoss.forward
        assert logits.ndim == 2
        assert target.ndim == 1
        assert logits.shape[-1] == len(self.margins)
        # mask[i,j] = 1 if target[i] == j else 0
        mask = torch.nn.functional.one_hot(target, num_classes=logits.shape[-1])
        new_logits = logits - self.margins.reshape(1, -1) * mask
        return F.cross_entropy(
            self.inv_temperature * new_logits,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )


class LogitAdjustedLoss(nn.Module):
    def __init__(
        self,
        weight: Optional[torch.Tensor],
        num_per_class: List[int],
        temperature: float,
        reduction: str,
    ):
        super().__init__()
        assert temperature > 0
        self.temperature = temperature
        self.reduction = reduction

        if num_per_class is None:
            raise ValueError(
                f"num_per_class must be set to initialize {self.__class__}"
            )
        num_per_class = np.array(num_per_class)
        adjustments = np.log(num_per_class / num_per_class.sum(0))
        adjustments = torch.as_tensor(adjustments, dtype=torch.get_default_dtype())
        self.register_buffer("adjustments", adjustments)
        self.register_buffer("weight", weight)
        if self.weight is not None:
            assert self.adjustments.shape == self.weight.shape

    def forward(self, logits, target):
        # follows the interface of torch.nn.CrossEntropyLoss.forward
        assert logits.ndim == 2
        assert target.ndim == 1
        assert logits.shape[-1] == len(self.adjustments)
        adjusted_logits = logits + self.temperature * self.adjustments.reshape(1, -1)
        return F.cross_entropy(
            adjusted_logits, target, weight=self.weight, reduction=self.reduction,
        )

class PolynomialLoss(nn.Module):
    """
    Poly-tailed margin based losses that decay as v^{-\alpha} for \alpha > 0.
    The theory here is that poly-tailed losses do not have max-margin behavior
    and thus can work with importance weighting.

    Poly-tailed losses are not defined at v=0 for v^{-\alpha}, and so there are
    several variants that are supported via the [[type]] option
    exp : f(v):= exp(-v+1) for v < 1, 1/v^\alpha otherwise
    logit: f(v):= 1/log(2)log(1+exp(-v+1)) for v < 1, 1/v^\alpha else.
    """
    allowed_types = {'exp', 'logit'}

    def __init__(self, type: str, alpha: float, reduction: str):
        super().__init__()
        self.type = type
        assert(type in self.allowed_types)
        assert(alpha > 0)
        self.alpha = alpha
        assert(reduction == "none")

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= 1
        scores = torch.zeros_like(margin_vals)
        inv_part = torch.pow(margin_vals, -1*self.alpha)
        if self.type == 'exp':
            exp_part = torch.exp(-1*margin_vals+1)
            scores[indicator] = exp_part[indicator]
            scores[~indicator] = inv_part[~indicator]
            return scores
        if self.type == 'logit':
            logit_inner = -1*margin_vals+1
            logit_part = (torch.log(torch.exp(-1*logit_inner)+1)+logit_inner)/math.log(2)
            scores[indicator] = logit_part[indicator]
            scores[~indicator] = inv_part[~indicator]
            print(margin_vals)
            return scores

    def forward(self, logits, target):
        """
        interfaces matches that of [[torch.nn.CrossEntropyLoss]] but treats
        the first coordinate of logits ([:,0]) as the binary classification score
        and discards all others
        """
        target_sign = 2*target-1
        margin_scores = logits[:,0]*target_sign
        return self.margin_fn(margin_scores)



