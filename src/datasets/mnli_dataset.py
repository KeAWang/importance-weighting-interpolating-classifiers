from datasets import Dataset as HFDataset
from .utils import GroupedLabeledDatapoint
from torch.utils.data import Dataset
from typing import Union
import torch
import numpy as np


class MNLIDataset(Dataset):
    groupby = "genre"

    def __init__(self, mnli_dataset: HFDataset, input_transform, frac=1.0):
        num_original = len(mnli_dataset)
        num_new = int(frac * num_original)
        rng = np.random.RandomState(
            0
        )  # always use the same seed for subsampling dataset
        chosen_indices = rng.choice(num_new, (num_new,), replace=False).tolist()
        chosen_mnli_dataset = mnli_dataset[chosen_indices]

        self.mnli_dataset = chosen_mnli_dataset
        self.input_transform = input_transform

        self.y_array = np.array(self.mnli_dataset["label"])

        group_labels = self.mnli_dataset[self.groupby]
        groups = sorted(list(set(group_labels)))
        self.idx_to_group = groups
        self.group_to_idx = {g: i for i, g in enumerate(groups)}
        self.group_array = np.array(list(self.group_to_idx[g] for g in group_labels))

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx: Union[torch.Tensor, int]):
        if torch.is_tensor(idx):
            idx = idx.item()
        premise = self.mnli_dataset["premise"][idx]
        hypothesis = self.mnli_dataset["hypothesis"][idx]
        x = self.input_transform((premise, hypothesis))
        y = self.y_array[idx]
        g = self.group_array[idx]
        return GroupedLabeledDatapoint(x=x, y=y, g=g)
