from datasets import Dataset as HFDataset
from .utils import GroupedLabeledDatapoint
from torch.utils.data import Dataset
import numpy as np


class MNLIDataset(Dataset):
    groupby = "genre"

    def __init__(self, mnli_dataset: HFDataset, input_transform):
        self.mnli_dataset = mnli_dataset
        self.input_transform = input_transform

        self.y_array = np.array(self.mnli_dataset["label"])

        group_labels = mnli_dataset[self.groupby]
        groups = sorted(list(set(group_labels)))
        self.idx_to_group = groups
        self.group_to_idx = {g: i for i, g in enumerate(groups)}
        self.group_array = np.array([self.group_to_idx[g] for g in group_labels])

    def __len__(self):
        return len(self.mnli_dataset)

    def __getitem__(self, idx):
        example = self.mnli_dataset[idx]
        x = self.input_transform(example)
        y = self.y_array[idx]
        g = self.group_array[idx]
        return GroupedLabeledDatapoint(x=x, y=y, g=g)
