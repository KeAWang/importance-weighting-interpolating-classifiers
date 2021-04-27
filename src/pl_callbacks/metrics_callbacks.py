from pytorch_lightning.callbacks.base import Callback
from typing import List  # Union, Dict
from pytorch_lightning.core.step_result import Result
from collections import defaultdict
import torch


class GroupAccuracyMonitor(Callback):
    """ Log at the end of each epoch
    """

    def __init__(self):
        self.group_pred_stats = {}
        self.reset("train")
        self.reset("val")
        return

    # Once pytorch lightning 1.3 is stable, we can use this below to replace all other
    # code. Since then we can get all outputs in on_validaton_epoch_end
    # def _on_epoch_end(
    #    self, trainer, pl_module, outputs: List[List[List[Result]]], mode
    # ):
    #    # outputs is a [optimizer outs][batch outs][tbptt steps] nested list
    #    # containing a dict for each batch of data
    #    # See pytorch_lightning.trainer.training_loop.TrainLoop._prepare_outputs from
    #    # newer version (1.3) of pytorch lightning for some more info

    #    # TODO: remove this when we upgrade to pytorch_lightning 1.3
    #    # See: https://github.com/PyTorchLightning/pytorch-lightning/pull/6969
    #    outputs: Dict[List] = _to_dict_of_lists(outputs)
    #    other: List = outputs["other"]
    #    # concatenate saved tensors from each batch
    #    group_labels = torch.cat(
    #        [o[0] for o in other]
    #    )  # NOTE: assume group label is first in `other`
    #    class_labels = torch.cat(outputs["targets"])
    #    pred_labels = torch.cat(outputs["preds"])

    #    correct = (class_labels == pred_labels).float()

    #    # Groupby: https://twitter.com/jeremyphoward/status/1185062637341593600
    #    idxs, vals = torch.unique(group_labels, return_counts=True)
    #    vs = torch.split_with_sizes(correct, tuple(vals))
    #    group_accs = {k.item(): v.mean().item() for k, v in zip(idxs, vs)}
    #    for group, acc in group_accs.items():
    #        pl_module.log(
    #            f"{mode}/group_{group}_acc",
    #            acc,
    #            on_step=False,
    #            on_epoch=True,
    #            prog_bar=False,
    #        )

    def reset(self, mode):
        self.group_pred_stats[mode] = defaultdict(lambda: (0, 0))

    def update(self, group_pred_stats, mode):
        for k, (num_correct, num_seen) in group_pred_stats.items():
            new_num_correct = self.group_pred_stats[mode][k][0] + num_correct
            new_num_seen = self.group_pred_stats[mode][k][1] + num_seen
            self.group_pred_stats[mode][k] = (new_num_correct, new_num_seen)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: List[List[List[Result]]],
        batch,
        batch_idx,
        dataloder_idx,
    ):
        self._on_batch_end(trainer, pl_module, outputs, mode="train")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: List[List[List[Result]]],
        batch,
        batch_idx,
        dataloder_idx,
    ):
        self._on_batch_end(trainer, pl_module, outputs, mode="val")

    def _on_batch_end(
        self, trainer, pl_module, outputs: List[List[List[Result]]], mode
    ):
        if mode == "train":
            assert len(outputs) == 1
            assert len(outputs[0]) == 1
            outputs = outputs[0][0]["extra"]
        group_labels = outputs["other"][0]  # NOTE: assume group label is first
        pred_labels = outputs["preds"]
        true_labels = outputs["targets"]

        correct = (pred_labels == true_labels).float()
        # Groupby: https://twitter.com/jeremyphoward/status/1185062637341593600
        idxs, vals = torch.unique(group_labels, return_counts=True)
        vs = torch.split_with_sizes(correct, tuple(vals))
        # Compute group: (num_correct, total_seen)
        group_pred_stats = {
            k.item(): (v.sum().item(), len(v)) for k, v in zip(idxs, vs)
        }
        self.update(group_pred_stats, mode=mode)

    def on_train_epoch_end(self, trainer, pl_module, outputs: List[List[List[Result]]]):
        self._on_epoch_end(trainer, pl_module, outputs, mode="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, None, mode="val")

    def _on_epoch_end(
        self, trainer, pl_module, outputs: List[List[List[Result]]], mode
    ):
        # For each group, average accuracy across all batches in epoch
        group_accs = {
            g: num_correct / num_seen
            for g, (num_correct, num_seen) in self.group_pred_stats[mode].items()
        }
        for group, acc in group_accs.items():
            pl_module.log(
                f"{mode}/group_{group}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
        self.reset(mode)


# def _to_dict_of_lists(list_of_dicts):
#    return {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0]}
