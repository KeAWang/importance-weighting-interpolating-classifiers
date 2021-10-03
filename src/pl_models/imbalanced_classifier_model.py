from typing import Any, Dict, List, Sequence, Tuple, Union, Optional
from copy import deepcopy
from math import ceil

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer


class ImbalancedClassifierModel(LightningModule):
    def __init__(
        self,
        architecture: torch.nn.Module,
        optimizer_config: DictConfig,
        lr_scheduler_config: Optional[DictConfig],
        loss_fn: DictConfig,
        freeze_features: bool,
        ckpt_path: Optional[str],
        output_size: int,
        dro: bool,
        adv_probs_lr: Optional[float],
        reweight_loss: bool,
        **unused_kwargs,
    ):
        super().__init__()
        print(
            f"{self.__class__} initialized with unused kwargs: {list(unused_kwargs.keys())}"
        )

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        # self.save_hyperparameters()
        # However this will cause the pytorch lightning saved state_dict to contain a
        # field called "hyper_parameters" which will be a copy of everything passed into
        # the LightningModule. This is not good for example, if we pass in an
        # architecture since it adds extra memory costs and requires the class
        # definition for the architecture when loading the checkpoint with torch.load

        self.lr_scheduler_config = lr_scheduler_config
        self.optimizer_config = optimizer_config

        self.architecture = architecture

        self.loss_fn = hydra.utils.instantiate(loss_fn, reduction="none")

        # Replace linear layer with new linear layer with possibly differ number of classes
        in_features = self.architecture.linear_output.in_features
        new_out = output_size
        self.architecture.linear_output.out_features = new_out
        print("Resetting final linear layer")
        self.architecture.linear_output.weight = torch.nn.Parameter(
            torch.Tensor(new_out, in_features)
        )
        self.architecture.linear_output.bias = torch.nn.Parameter(torch.Tensor(new_out))
        self.architecture.linear_output.reset_parameters()

        if ckpt_path is not None:
            print(f"Loading from checkpoint path {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            self.architecture.load_state_dict(
                submodule_state_dict(ckpt["state_dict"], "architecture")
            )

        self.freeze_features = freeze_features
        if freeze_features:
            print("Freezing feature extractor")
            set_grad(self.architecture, requires_grad=False)
            # Remember to set non-classifier layers to eval mode in self.train()!!!
        else:
            set_grad(self.architecture, requires_grad=True)
        set_grad(self.architecture.linear_output, requires_grad=True)

        self.dro = dro
        self.adv_probs: Optional[
            torch.Tensor
        ] = None  # will be populated later if dro is True
        if dro:
            assert adv_probs_lr is not None
        self.adv_probs_lr = adv_probs_lr

        self.reweight_loss = reweight_loss

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def train(self, mode: bool = True):
        if mode:
            super().train(True)
            # make sure things like batch norm are actually frozen
            if self.freeze_features:
                self.architecture.eval()
                self.architecture.linear_output.train()
        else:
            super().train(False)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=trainable_parameters(self)
        )
        if self.lr_scheduler_config is not None:
            interval = self.lr_scheduler_config["interval"]
            del self.lr_scheduler_config["interval"]
            if self.lr_scheduler_config.get("num_training_steps") == "DYNAMIC":
                num_train = len(self.trainer.datamodule.train_dataset)
                steps_per_epoch = ceil(num_train / self.trainer.datamodule.batch_size)
                num_epochs = self.trainer.max_epochs
                self.lr_scheduler_config["num_training_steps"] = (
                    num_epochs * steps_per_epoch
                )

            lr_scheduler = hydra.utils.instantiate(
                self.lr_scheduler_config, optimizer=optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": interval},
            }

        return optimizer

    def forward(self, x) -> torch.Tensor:
        return self.architecture(x)

    def step(
        self, batch: Tuple, training: bool = True
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]
    ]:
        if isinstance(batch, tuple) and hasattr(
            batch, "_fields"
        ):  # check if namedtuple
            x, y, w, g = batch.x, batch.y, batch.w, batch.g
            other_data = batch._asdict()
            other_data.pop("x")
            other_data.pop("y")
        else:
            x, y = batch
            w = torch.ones_like(y)
            other_data = {}

        targets = y

        if self.dro:
            if self.adv_probs is None:
                # Hacky way of getting adv_probs without needing to know number of
                # groups at __init__ time since datamodule is not initialiezd yet at
                # __init__
                num_groups = len(self.trainer.datamodule.train_g_counter.keys())
                adv_probs = torch.ones(num_groups, device=x.device, dtype=x.dtype)
                del self.adv_probs
                self.register_buffer("adv_probs", adv_probs)

        # In case of batch norm, need to be careful with .eval() and .train()
        with torch.no_grad():
            self.eval()
            preds = torch.argmax(self(x), dim=-1)

            self.train(training)

        logits = self(x)
        losses = self.loss_fn(logits, y)

        if self.dro:
            assert g.ndim == 1
            with torch.no_grad():
                self.eval()
                group_losses = compute_avg_group_losses(losses, g, len(self.adv_probs))
                adv_probs = self.adv_probs
                adv_probs = adv_probs * torch.exp(self.adv_probs_lr * group_losses)
                adv_probs = adv_probs / adv_probs.sum(0)
                self.adv_probs = adv_probs

                self.train(training)
            losses = losses * self.adv_probs[g]

        if self.reweight_loss:
            losses = losses * w

        return losses, logits, preds, targets, other_data

    # See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks for hooks order
    # Newest hooks are in https://github.com/PyTorchLightning/pytorch-lightning/pull/7713

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
    #    breakpoint()

    # def on_train_epoch_start(self):
    #    breakpoint()

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        losses, logits, preds, targets, other_data = self.step(batch, training=True)
        loss = losses.mean(0)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {
            "loss": loss,
            "losses": losses,
            "logits": logits,
            "preds": preds,
            "targets": targets,
            **other_data,
        }

    def training_epoch_end(self, outputs: List[Any]) -> None:
        """
        outputs: list of dictionaries from self.training_step, one entry for each step within epoch

        """
        # log best so far train acc and train loss
        # TODO: what is self.trainer.callback_metrics?
        trainer: Trainer = self.trainer
        self.metric_hist["train/acc"].append(trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        losses, logits, preds, targets, other_data = self.step(batch, training=False)
        loss = losses.mean(0)

        num_examples = len(targets)
        num_pos_pred = (preds == 1).sum().item()

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "losses": losses,
            "logits": logits,
            "preds": preds,
            "targets": targets,
            "num_examples": num_examples,
            "num_pos_pred": num_pos_pred,
            **other_data,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        trainer: Trainer = self.trainer
        self.metric_hist["val/acc"].append(trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

        num_examples = sum([o["num_examples"] for o in outputs])
        num_pos_pred = sum([o["num_pos_pred"] for o in outputs])
        frac_predicted_pos = num_pos_pred / num_examples
        self.log("val/frac_predicted_pos", frac_predicted_pos, prog_bar=False)

    def test_step(self, batch: Any, batch_idx: int):
        losses, logits, preds, targets, other_data = self.step(batch, training=False)
        loss = losses.mean(0)

        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
            "targets": targets,
            **other_data,
        }


def set_grad(module: torch.nn.Module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def match_flattened_params(module1, module2):
    params1 = dict(module1.named_parameters())
    params2 = dict(module2.named_parameters())
    assert set(params1.keys()) == set(params2.keys())
    p_list1 = []
    p_list2 = []
    for n in params1:
        p1 = params1[n].flatten()
        p2 = params2[n].flatten()
        p_list1.append(p1)
        p_list2.append(p2)
    return torch.cat(p_list1), torch.cat(p_list2)


def submodule_state_dict(state_dict: Dict[str, torch.Tensor], submodule_name: str):
    assert len(submodule_name) > 0
    # keep only attributes of submodule and remove prefix
    def remove_prefix(s, p):
        return s[len(p) :]

    submodule_state_dict = {
        remove_prefix(k, f"{submodule_name}."): v
        for k, v in state_dict.items()
        if k.startswith(f"{submodule_name}.")
    }
    return submodule_state_dict


def trainable_parameters(module: torch.nn.Module):
    # otherwise things like adam might change params even with requires_grad=False
    # See https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/4
    for p in module.parameters():
        if p.requires_grad:
            yield p


def trainable_named_parameters(module: torch.nn.Module):
    # otherwise things like adam may change params even with requires_grad=False
    # See https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/4
    for n, p in module.named_parameters():
        if p.requires_grad:
            yield n, p


def groupby_mean(
    values: torch.Tensor, labels: torch.LongTensor
) -> Tuple[torch.Tensor, torch.LongTensor]:
    """Group-wise average for (sparse) grouped tensors
    Modified from https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/9

    Args:
        values (torch.Tensor): values to average (# samples, latent dimension)
        labels (torch.LongTensor): labels for embedding parameters (# samples,)

    Returns:
        result (torch.Tensor): (# unique labels, latent dimension)
        new_labels (torch.LongTensor): (# unique labels,)

    Examples:
        >>> samples = torch.Tensor([
                             [0.15, 0.15, 0.15],    #-> group / class 1
                             [0.2, 0.2, 0.2],    #-> group / class 5
                             [0.4, 0.4, 0.4],    #-> group / class 5
                             [0.0, 0.0, 0.0]     #-> group / class 0
                      ])
        >>> labels = torch.LongTensor([1, 5, 5, 0])
        >>> result, new_labels = groupby_mean(samples, labels)

        >>> result
        tensor([[0.0000, 0.0000, 0.0000],
            [0.1500, 0.1500, 0.1500],
            [0.3000, 0.3000, 0.3000]])

        >>> new_labels
        tensor([0, 1, 5])
    """
    one_dim = values.ndim == 1
    if one_dim:
        values = values.unsqueeze(-1)
    assert values.ndim == 2
    assert labels.ndim == 1
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.tensor(
        list(map(key_val.get, labels)), dtype=torch.long, device=values.device
    )

    labels = labels.view(labels.size(0), 1).expand(-1, values.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=values.dtype).scatter_add_(
        0, labels, values
    )
    result = result / labels_count.float().unsqueeze(1)
    new_labels = torch.tensor(
        list(map(val_key.get, unique_labels[:, 0].tolist())),
        dtype=torch.long,
        device=values.device,
    )
    if one_dim:
        result = result.squeeze(-1)
    return result, new_labels


def get_worst_group_idx(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    assert weights.ndim == 1
    unique_w = weights.unique()
    w_onehot = weights.unsqueeze(-1) == unique_w
    labels = w_onehot.nonzero()[:, -1]

    assert losses.ndim == 1
    group_losses, new_labels = groupby_mean(losses, labels)
    worst_group = new_labels[group_losses.argmax()]

    worst_group_idx = (labels == worst_group).nonzero().squeeze(-1)
    return worst_group_idx


def compute_avg_group_losses(
    losses: torch.Tensor, group_idx: torch.Tensor, num_groups: int
):
    # compute observed counts and mean loss for each group
    assert losses.ndim == 1
    assert group_idx.ndim == 1
    group_map = (
        torch.arange(num_groups, device=losses.device, dtype=losses.dtype).unsqueeze(1)
        == group_idx
    ).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()  # prevent division by zero
    group_loss = (group_map @ losses) / group_denom
    return group_loss
