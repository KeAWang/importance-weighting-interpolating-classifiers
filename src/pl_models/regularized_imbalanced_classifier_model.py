from typing import Any, Dict, List, Sequence, Tuple, Union, Optional
from copy import deepcopy

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer


class RegularizedImbalancedClassifierModel(LightningModule):
    def __init__(
        self,
        architecture: torch.nn.Module,
        optimizer: DictConfig,
        loss_fn: DictConfig,
        freeze_features: bool,
        ckpt_path: Optional[str],
        output_size: int,
        reference_regularization: float,
        regularization_type: str,
        reinit: bool,
        only_update_misclassified: bool,
        **unused_kwargs,
    ):
        super().__init__()
        print(
            f"{self.__class__} initialized with unused kwargs: {list(unused_kwargs.keys())}"
        )

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

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

        old_state_dict = deepcopy(self.architecture.state_dict())
        # Load weights if needed
        if ckpt_path is not None:
            print(f"Loading from checkpoint path {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            self.load_state_dict(ckpt["state_dict"])

        if freeze_features:
            print("Freezing feature extractor")
            set_grad(self.architecture, requires_grad=False)
        else:
            set_grad(self.architecture, requires_grad=True)
        set_grad(self.architecture.linear_output, requires_grad=True)

        self.reference_architecture = deepcopy(self.architecture)
        set_grad(self.reference_architecture, requires_grad=False)
        self.reference_regularization = reference_regularization

        self.regularization_type = regularization_type

        if reinit:
            self.architecture.load_state_dict(old_state_dict)
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.only_update_misclassified = only_update_misclassified

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        return optim

    def forward(self, x) -> torch.Tensor:
        return self.architecture(x)

    def step(
        self, batch: Tuple
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
        logits, reference_logits = self.forward(x), self.reference_architecture(x)
        preds = torch.argmax(logits, dim=-1)
        loss = self.loss_fn(logits, y)
        # TODO: do we need to change the normalization?
        if self.only_update_misclassified:
            # This will rescale to reflect the changed effective batch size
            correct = preds != y
            w = w * correct.float()
            reweighted_loss = (loss * w).sum(0) / (w.sum(0) + 1e-5)

            # This will not rescale
            # reweighted_loss = (loss * w * (preds != y).float()).sum(0) / w.sum(0)
        else:
            reweighted_loss = (loss * w).sum(0) / w.sum(0)

        if self.regularization_type == "logits":
            kl = (
                -(reference_logits.softmax(dim=-1) * logits.log_softmax(dim=-1))
                .sum(-1)
                .mean(0)
            )
            reg_term = kl
        elif self.regularization_type == "weights":
            learned_params, ref_params = match_flattened_params(
                self.architecture, self.reference_architecture
            )
            l2 = (learned_params - ref_params).pow(2).sum(0)
            reg_term = l2
        total_loss = reweighted_loss + self.reference_regularization * reg_term
        return total_loss, logits, preds, y, other_data

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, logits, preds, targets, other_data = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
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
        loss, logits, preds, targets, other_data = self.step(batch)
        num_examples = len(targets)
        num_pos_pred = (preds == 1).sum().item()

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
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
