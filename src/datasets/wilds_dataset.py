from torch.utils.data import Dataset
import wilds
from wilds.common.grouper import CombinatorialGrouper
from .utils import GroupedLabeledDatapoint
from functools import cached_property


class WILDSDataset(Dataset):
    def __init__(
        self,
        wilds_dataset: wilds.datasets.wilds_dataset.WILDSDataset,
        grouper: CombinatorialGrouper,
    ):
        ## TODO: transforms?
        self.wilds_dataset = wilds_dataset
        self.grouper = grouper

    def __len__(self):
        return len(self.wilds_dataset)

    def __getitem__(self, idx):
        x, y, _ = self.wilds_dataset[idx]
        g = self.group_array[idx]
        return GroupedLabeledDatapoint(x=x, y=y, g=g)

    @cached_property
    def y_array(self):
        return self.wilds_dataset.y_array

    @cached_property
    def group_array(self):
        return self.grouper.metadata_to_group(self.wilds_dataset.metadata_array)
