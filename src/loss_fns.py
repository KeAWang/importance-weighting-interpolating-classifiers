import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class LDAMLoss(nn.Module):
    """Reimplementation of https://github.com/kaidic/LDAM-DRW/blob/master/losses.py

    The original published code included a temperature hyperparameter, named `s` in the
    released code, that is by default set to 30. This was not mentioned in the paper.

    """

    def __init__(
        self,
        num_per_class: List[int],
        max_margin: float,
        inv_temperature: float,
        reduction: str = "none",
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
        self.reduction = reduction

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
            adjusted_logits,
            target,
            weight=self.weight,
            reduction=self.reduction,
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

    allowed_types = {"exp", "logit", "linear"}

    def __init__(self, type: str, alpha: float, reduction: str):
        super().__init__()
        self.type = type
        assert type in self.allowed_types
        self.alpha = float(alpha)
        assert reduction == "none"

    def margin_fn(self, margin_vals: torch.Tensor):
        indicator = margin_vals <= 1
        inv_part = torch.pow(
            margin_vals.abs(), -1 * self.alpha
        )  # prevent exponentiating negative numbers by fractional powers
        if self.type == "exp":
            exp_part = torch.exp(-1 * margin_vals)
            scores = exp_part * indicator + inv_part * (~indicator)
            return scores
        if self.type == "logit":
            indicator = margin_vals <= 1
            inv_part = torch.pow(margin_vals.abs(), -1 * self.alpha)
            logit_inner = -1 * margin_vals
            logit_part = torch.nn.functional.softplus(logit_inner) / (
                math.log(1 + math.exp(-1))
            )
            scores = logit_part * indicator + inv_part * (~indicator)
            return scores
        if self.type == "linear":
            assert self.alpha > 1
            linear_part = -1 * margin_vals + torch.ones_like(margin_vals) * (
                self.alpha / (self.alpha - 1)
            )
            scores = linear_part * indicator + inv_part * (~indicator) / (
                self.alpha - 1
            )
            return scores

    def forward(self, logits, target):
        target_sign = 2 * target - 1
        margin_scores = (logits[:, 1] - logits[:, 0]) * target_sign
        loss_values = self.margin_fn(margin_scores)
        return loss_values


class VSLoss(nn.Module):
    def __init__(
        self,
        tau: float,
        gamma: float,
        num_per_class: List[int],
        reduction: str = "none",
    ):
        """Note this only works for class imbalance and not group imbalance for now"""
        super().__init__()
        num_per_class = torch.tensor(num_per_class, dtype=torch.long)
        pi = num_per_class / num_per_class.sum(0)  # prior probabilities
        additive_param = tau * torch.log(pi)  # iota

        n_max = torch.max(num_per_class).item()
        multiplicative_param = (num_per_class / n_max) ** gamma  # delta

        self.tau = tau
        self.gamma = gamma
        self.num_per_class = num_per_class
        self.reduction = reduction

        self.register_buffer("additive_param", additive_param)
        self.register_buffer("multiplicative_param", multiplicative_param)

    def forward(self, logits, target):
        assert logits.ndim == 2
        new_logits = self.multiplicative_param * logits + self.additive_param
        loss = F.cross_entropy(new_logits, target, reduction=self.reduction)
        return loss


class VSGroupLoss(nn.Module):
    _takes_groups = True

    def __init__(
        self,
        gamma: float,
        num_per_group: List[int],
        reduction: str = "none",
    ):
        super().__init__()
        num_per_group = torch.tensor(num_per_group, dtype=torch.long)

        n_max = torch.max(num_per_group).item()
        additive_param = -1 / ((num_per_group / n_max) ** gamma)  # iota
        multiplicative_param = (num_per_group / n_max) ** gamma  # delta

        self.gamma = gamma
        self.num_per_group = num_per_group
        self.reduction = reduction

        self.register_buffer("additive_param", additive_param)
        self.register_buffer("multiplicative_param", multiplicative_param)

    def forward(self, logits, target, groups):
        assert logits.ndim == 2
        new_logits = self.multiplicative_param[groups].reshape(
            -1, 1
        ) * logits + self.additive_param[groups].reshape(-1, 1)
        loss = F.cross_entropy(new_logits, target, reduction=self.reduction)
        return loss
