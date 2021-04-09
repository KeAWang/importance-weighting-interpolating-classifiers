from typing import List
import torch
import numpy as np


class ImbalancedDatasetMixin:
    def compute_weights(self, targets, target_weights="balanced"):
        """
        compute_weights(100, 200]) should return [2/3, 1/3]

        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(targets)
        class_weights = compute_class_weight(
            target_weights, classes=classes, y=targets,
        )  # unnormalized weights
        return class_weights

    def remap_labels(self, targets, desired_classes: List[int]):
        # targets should not have any labels belonging to the other classes
        assert len(set(targets.tolist()) - set(desired_classes)) == 0

        # old_to_new[i] is the new label of class i
        old_to_new = np.arange(10)
        old_to_new[desired_classes] = np.arange(len(desired_classes))
        other_classes = list(set(range(10)) - set(desired_classes))
        old_to_new[other_classes] = np.arange(len(desired_classes), 10)
        # transform old labels to new labels, e.g. class 4,6,9 become class
        # 0,1,2 while class 0, 1, 2, 3, 5, 7, 8 become class 3 to 9
        new_targets = old_to_new[targets]

        return new_targets

    def resample_dataset(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        desired_classes: List[int],
        num_undersample_per_class: List[int],
        num_oversample_per_class: List[int],
    ):
        # TODO: handle when num*sample is None
        num_undersample_per_class = dict(
            zip(desired_classes, num_undersample_per_class)
        )
        num_oversample_per_class = dict(zip(desired_classes, num_oversample_per_class))

        new_inputs = []
        new_targets = []
        for cls in desired_classes:
            num_undersample = num_undersample_per_class[cls]
            num_oversample = num_oversample_per_class[cls]
            assert targets.ndim == 1
            cls_idx = np.nonzero(targets == cls)[0]
            # Can't undersample more than number of samples in that class
            assert num_undersample <= len(
                cls_idx
            ), f"num_undersample: {num_undersample} > len(cls_idx): {len(cls_idx)}"
            # Can't oversample less than what we first undersample to
            assert num_oversample >= num_undersample

            # first we undersample
            cls_idx = cls_idx[
                :num_undersample
            ]  # always choose the first `num_undersample` to reduce variance when we change `num_undersample` in our experiments

            # now we oversample with replacement, while keeping everything we
            # previously subsampled
            num_remaining = num_oversample - len(cls_idx)
            # TODO: convert to just numpy
            additional_idx = torch.randint(
                0,
                len(cls_idx),
                (num_remaining,),
                generator=torch.Generator().manual_seed(
                    34028
                ),  # always use same seed for oversampling. 34028 is randomly chosen
            ).numpy()
            additional_idx = cls_idx[additional_idx]
            cls_idx = np.concatenate([cls_idx, additional_idx])

            new_inputs.append(inputs[cls_idx])
            new_targets.append(targets[cls_idx])

        new_inputs = np.concatenate(new_inputs)
        new_targets = np.concatenate(new_targets)

        return new_inputs, new_targets

    def select_classes(self, inputs, targets, desired_classes: List[int]):
        desired_classes = np.array(desired_classes)
        desired_idx = [
            i for i, target in enumerate(targets) if target in desired_classes
        ]
        desired_idx = np.array(desired_idx)

        # keep only examples with the desired labels
        new_targets = targets[desired_idx]
        new_inputs = inputs[desired_idx]

        return new_inputs, new_targets
