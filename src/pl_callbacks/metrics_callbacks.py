from collections import Counter
from pytorch_lightning.callbacks.base import Callback
from typing import List
import wandb
import torch


class GroupAccuracyMonitor(Callback):
    """Computes and logs accuracy of each group.

    Expects outputs to contain predictions, targets, and groups
    """

    def __init__(self, mode):
        self.mode = mode
        self.preds = []
        self.targets = []
        self.groups = []

        self.reset()

    def reset(self):
        self.preds.clear()
        self.targets.clear()
        self.groups.clear()

    def on_batch_end_shared(
        self, outputs,
    ):
        """Gather data from single batch."""
        self.preds.append(outputs["preds"])
        self.targets.append(outputs["targets"])
        self.groups.append(outputs["g"])

    def on_epoch_end_shared(self, pl_module):
        # concatenate saved tensors from each batch
        group_labels = torch.cat(self.groups)
        class_labels = torch.cat(self.targets)
        pred_labels = torch.cat(self.preds)

        correct = (class_labels == pred_labels).float()

        group_accs = {}
        for g in torch.unique(group_labels).tolist():
            in_g = group_labels == g
            acc = correct[in_g].mean().item()
            group_accs[g] = acc

            pl_module.log(
                f"{self.mode}/group_{g}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self.reset()


class GroupTrainAccuracyMonitor(GroupAccuracyMonitor):
    def __init__(self):
        super().__init__(mode="train")

    def on_train_batch_end(
        self, trainer, pl_module, outputs: List[List[dict]], *args, **kwargs
    ):
        assert len(outputs) == 1
        assert len(outputs[0]) == 1
        outputs = outputs[0][0]["extra"]
        self.on_batch_end_shared(outputs)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        self.on_epoch_end_shared(pl_module)


class GroupValAccuracyMonitor(GroupAccuracyMonitor):
    def __init__(self):
        super().__init__(mode="val")

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.on_batch_end_shared(outputs)

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        self.on_epoch_end_shared(pl_module)


class GroupTestAccuracyMonitor(GroupAccuracyMonitor):
    def __init__(self):
        super().__init__(mode="test")

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.on_batch_end_shared(outputs)

    def on_test_epoch_end(self, trainer, pl_module, *args, **kwargs):
        self.on_epoch_end_shared(pl_module)


class GroupValReweightedAccuracyMonitor(Callback):
    train_group_counts: Counter
    val_preds: list
    val_targets: list
    val_groups: list
    val_weights: list

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.train_group_counts = Counter()
        self.val_preds = []
        self.val_targets = []
        self.val_groups = []
        self.val_weights = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs: List[List[dict]], *args, **kwargs
    ):
        assert len(outputs) == 1
        assert len(outputs[0]) == 1
        outputs = outputs[0][0]["extra"]
        self.train_group_counts += Counter(outputs["g"].tolist())

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        """Gather data from single batch."""
        self.val_preds.append(outputs["preds"])
        self.val_targets.append(outputs["targets"])
        self.val_groups.append(outputs["g"])
        self.val_weights.append(outputs["w"])

    def on_validation_epoch_end(self, trainer, pl_module, *args, **kwargs):
        # concatenate saved tensors from each batch
        group_labels = torch.cat(self.val_groups)
        class_labels = torch.cat(self.val_targets)
        pred_labels = torch.cat(self.val_preds)
        weights = torch.cat(self.val_weights)

        correct = (class_labels == pred_labels).float()

        group_accs = {}
        worst_acc = float("inf")
        for g in torch.unique(group_labels).tolist():
            in_g = group_labels == g
            acc = correct[in_g].mean().item()
            group_accs[g] = acc
            worst_acc = min(worst_acc, acc)

            pl_module.log(
                f"val/group_{g}_acc", acc, on_step=False, on_epoch=True, prog_bar=False,
            )
        pl_module.log(
            "val/worst_group_acc",
            worst_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        val_reweighted_acc = (correct * weights).sum(0) / weights.sum(0)
        pl_module.log(
            "val/val_reweighted_acc",
            val_reweighted_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # compute val_acc reweighted by group counts in train set
        total_train = sum(self.train_group_counts.values())
        if total_train > 0:  # validation sanity checks won't have any training samples
            reweighted_acc = 0
            total_in_train_from_val_groups = 0  # sometimes we don't see every train group in validation set, e.g.  when we train for less than one epoch
            for g in group_accs:
                c = self.train_group_counts[g]
                reweighted_acc += group_accs[g] * c
                total_in_train_from_val_groups += c
            reweighted_acc /= total_in_train_from_val_groups

            print(f"Group counts in training set: {self.train_group_counts}")
            pl_module.log(
                "val/train_reweighted_acc",
                reweighted_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

        self.reset()


class GroupTestReweightedAccuracyMonitor(Callback):
    train_group_counts: Counter
    test_preds: list
    test_targets: list
    test_groups: list
    test_weights: list

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.train_group_counts = Counter()
        self.test_preds = []
        self.test_targets = []
        self.test_groups = []
        self.test_weights = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.reset()  # use counts from the most recent train epoch

    def on_train_batch_end(
        self, trainer, pl_module, outputs: List[List[dict]], *args, **kwargs
    ):
        assert len(outputs) == 1
        assert len(outputs[0]) == 1
        outputs = outputs[0][0]["extra"]
        self.train_group_counts += Counter(outputs["g"].tolist())

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        """Gather data from single batch."""
        self.test_preds.append(outputs["preds"])
        self.test_targets.append(outputs["targets"])
        self.test_groups.append(outputs["g"])
        self.test_weights.append(outputs["w"])

    def on_test_epoch_end(self, trainer, pl_module, *args, **kwargs):
        # concatenate saved tensors from each batch
        group_labels = torch.cat(self.test_groups)
        class_labels = torch.cat(self.test_targets)
        pred_labels = torch.cat(self.test_preds)
        weights = torch.cat(self.test_weights)

        correct = (class_labels == pred_labels).float()

        group_accs = {}
        worst_acc = float("inf")
        for g in torch.unique(group_labels).tolist():
            in_g = group_labels == g
            acc = correct[in_g].mean().item()
            group_accs[g] = acc
            worst_acc = min(worst_acc, acc)

            pl_module.log(
                f"test/group_{g}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        pl_module.log(
            "test/worst_group_acc",
            worst_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        test_reweighted_acc = (correct * weights).sum(0) / weights.sum(0)
        pl_module.log(
            "test/test_reweighted_acc",
            test_reweighted_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # compute test_acc reweighted by group counts in train set
        reweighted_acc = 0
        total_in_train_from_test_groups = 0  # sometimes we don't see every train group in test set, e.g.  when we train for less than one epoch
        for g in group_accs:
            c = self.train_group_counts[g]
            reweighted_acc += group_accs[g] * c
            total_in_train_from_test_groups += c
        reweighted_acc /= total_in_train_from_test_groups

        print(f"Group counts in training set: {self.train_group_counts}")
        pl_module.log(
            "test/train_reweighted_acc",
            reweighted_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )


class TrainLossHistogramMonitor(Callback):
    def __init__(self):
        self.losses = []

    def reset(self):
        self.losses.clear()

    def on_train_batch_end(
        self, trainer, pl_module, outputs: List[List[dict]], *args, **kwargs
    ):
        outputs = outputs[0][0]["extra"]
        self.losses += outputs["losses"].tolist()

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        # How to ensure we log every epoch? Difference between epoch and global step?
        trainer.logger.experiment[0].log({"train/losses": wandb.Histogram(self.losses)})
        self.reset()
