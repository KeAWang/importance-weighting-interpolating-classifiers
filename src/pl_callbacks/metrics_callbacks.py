from pytorch_lightning.callbacks.base import Callback
from typing import List, Union, Dict
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.core.step_result import Result
import torch


class GroupAccuracyMonitor(Callback):
    """ Log at the end of each epoch
    """

    def __init__(self):
        return

    def _on_epoch_end(
        self, trainer, pl_module, outputs: List[List[List[Result]]], mode
    ):
        # outputs is a [optimizer outs][batch outs][tbptt steps] nested list
        # containing a dict for each batch of data
        # See pytorch_lightning.trainer.training_loop.TrainLoop._prepare_outputs from
        # newer version (1.3) of pytorch lightning for some more info

        # TODO: remove this when we upgrade to pytorch_lightning 1.3
        # See: https://github.com/PyTorchLightning/pytorch-lightning/pull/6969
        # outputs: List[Dict] = _prepare_outputs(outputs, batch_mode=False)
        outputs: Dict[List] = _to_dict_of_lists(outputs)
        other: List = outputs["other"]
        # concatenate saved tensors from each batch
        group_labels = torch.cat(
            [o[0] for o in other]
        )  # NOTE: assume group label is first in `other`
        class_labels = torch.cat(outputs["targets"])
        pred_labels = torch.cat(outputs["preds"])

        correct = (class_labels == pred_labels).float()

        # Groupby: https://twitter.com/jeremyphoward/status/1185062637341593600
        idxs, vals = torch.unique(group_labels, return_counts=True)
        vs = torch.split_with_sizes(correct, tuple(vals))
        group_accs = {k.item(): v.mean().item() for k, v in zip(idxs, vs)}
        for group, acc in group_accs.items():
            pl_module.log(
                f"{mode}/group_{group}_acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def on_train_epoch_end(self, trainer, pl_module, outputs: List[List[List[Result]]]):
        self._on_epoch_end(trainer, pl_module, outputs, mode="train")

    def on_validation_epoch_end(
        self, trainer, pl_module, outputs: List[List[List[Result]]]
    ):
        self._on_epoch_end(trainer, pl_module, outputs, mode="val")


def _prepare_outputs(
    outputs: List[List[List[Result]]], batch_mode: bool,
) -> Union[List[List[List[Dict]]], List[List[Dict]], List[Dict], Dict]:
    """
    Extract required information from batch or epoch end results.
    Args:
        outputs: A 3-dimensional list of ``Result`` objects with dimensions:
            [optimizer outs][batch outs][tbptt steps].
        batch_mode: If True, ignore the batch output dimension.
    Returns:
        The cleaned outputs with ``Result`` objects converted to dictionaries. All list dimensions of size one will
        be collapsed.
    """
    processed_outputs = []
    for opt_outputs in outputs:
        # handle an edge case where an optimizer output is the empty list
        if len(opt_outputs) == 0:
            continue

        processed_batch_outputs = []

        if batch_mode:
            opt_outputs = [opt_outputs]

        for batch_outputs in opt_outputs:
            processed_tbptt_outputs = []

            for tbptt_output in batch_outputs:
                out = tbptt_output.extra
                out["loss"] = tbptt_output.minimize
                processed_tbptt_outputs.append(out)

            # if there was only one tbptt step then we can collapse that dimension
            if len(processed_tbptt_outputs) == 1:
                processed_tbptt_outputs = processed_tbptt_outputs[0]
            processed_batch_outputs.append(processed_tbptt_outputs)

        # batch_outputs should be just one dict (or a list of dicts if using tbptt) per optimizer
        if batch_mode:
            processed_batch_outputs = processed_batch_outputs[0]
        processed_outputs.append(processed_batch_outputs)

    # if there is only one optimiser then we collapse that dimension
    if len(processed_outputs) == 1:
        processed_outputs = processed_outputs[0]
    return processed_outputs


def _to_dict_of_lists(list_of_dicts):
    return {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0]}
