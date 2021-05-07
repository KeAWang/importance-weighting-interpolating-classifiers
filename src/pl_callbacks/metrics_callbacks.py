from pytorch_lightning.callbacks.base import Callback
from typing import List
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

        # Groupby: https://twitter.com/jeremyphoward/status/1185062637341593600
        sorted_group_labels = group_labels.sort().values
        idxs, vals = torch.unique(sorted_group_labels, sorted=True, return_counts=True)
        vs = torch.split_with_sizes(correct, tuple(vals))
        group_accs = {k.item(): v.mean().item() for k, v in zip(idxs, vs)}
        for group, acc in group_accs.items():
            pl_module.log(
                f"{self.mode}/group_{group}_acc",
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
