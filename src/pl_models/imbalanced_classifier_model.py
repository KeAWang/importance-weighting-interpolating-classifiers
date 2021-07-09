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
        reference_ckpt_path: Optional[str],
        output_size: int,
        reference_regularization: float,
        dont_update_correct_extras: bool,
        regularization_type: Optional[str],
        flood_level: Optional[float],
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

        self.reference_regularization = reference_regularization
        self.regularization_type = regularization_type
        if regularization_type is None:
            self.reference_architecture = None
        else:
            self.reference_architecture = deepcopy(self.architecture)
            set_grad(self.reference_architecture, requires_grad=False)
            if reference_ckpt_path is None and ckpt_path is not None:
                reference_ckpt_path = ckpt_path
            if reference_ckpt_path is not None:
                print(f"Loading from reference checkpoint path {reference_ckpt_path}")
                reference_ckpt = torch.load(
                    reference_ckpt_path, map_location=torch.device("cpu")
                )
                self.reference_architecture.load_state_dict(
                    submodule_state_dict(reference_ckpt["state_dict"], "architecture")
                )
            else:
                raise RuntimeError("No checkpoints or reference checkpoints given")

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

        self.dont_update_correct_extras = dont_update_correct_extras

        self.flood_level = flood_level

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
            if self.reference_architecture is not None:
                self.reference_architecture.eval()
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
            x, y, w = batch.x, batch.y, batch.w
            other_data = batch._asdict()
            other_data.pop("x")
            other_data.pop("y")
        else:
            x, y = batch
            w = torch.ones_like(y)
            other_data = {}

        with torch.no_grad():
            self.eval()
            preds = torch.argmax(self(x), dim=-1)
            self.train(training)

        # TODO: do we need to change the normalization?
        if training and self.dont_update_correct_extras:
            extra = batch.extra
            correct = preds == y
            # We need to prevent the model from even "seeing" the correct extra examples
            # since otherwise the batch norm statistics will change
            subset = ~(extra & correct)
            if not torch.any(subset):
                reweighted_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                logits = None
            else:
                x_ = x[subset]
                y_ = y[subset]
                w_ = w[subset]
                logits = self(x_)
                loss = self.loss_fn(logits, y_)
                reweighted_loss = (loss * w_).sum(0) / (
                    w_.sum(0)
                )  # prevent denom from blowing up when we get everything right
        else:
            logits = self(x)
            loss = self.loss_fn(logits, y)
            if self.flood_level is not None:
                loss = (loss - self.flood_level).abs() + self.flood_level
            reweighted_loss = (loss * w).sum(0) / w.sum(0)

        if self.regularization_type == "logits":
            reference_logits = self.reference_architecture(x)
            kl = (
                -(reference_logits.softmax(dim=-1) * logits.log_softmax(dim=-1))
                .sum(-1)
                .mean(0)
            )
            reg_term = self.reference_regularization * kl
        elif self.regularization_type == "weights":
            learned_params, ref_params = match_flattened_params(
                self.architecture, self.reference_architecture
            )
            l2 = (learned_params - ref_params).pow(2).sum(0)
            reg_term = self.reference_regularization * l2
        elif self.regularization_type is None:
            reg_term = 0
        total_loss = reweighted_loss + reg_term
        return (total_loss, reweighted_loss, reg_term), logits, preds, y, other_data

    # See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks for hooks order
    # Newest hooks are in https://github.com/PyTorchLightning/pytorch-lightning/pull/7713

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
    #    breakpoint()

    # def on_train_epoch_start(self):
    #    breakpoint()

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        losses, logits, preds, targets, other_data = self.step(batch, training=True)
        loss, ce_term, reg_term = losses

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/ce_term", ce_term, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/reg_term", reg_term, on_step=False, on_epoch=True, prog_bar=False
        )
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {
            "loss": loss,
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
        loss, ce_term, reg_term = losses

        num_examples = len(targets)
        num_pos_pred = (preds == 1).sum().item()

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/ce_term", ce_term, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/reg_term", reg_term, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
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
        loss, ce_term, reg_term = losses

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
