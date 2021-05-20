from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from collections import Counter
import torch


class BaseDataModule(LightningDataModule):
    """
    A DataModule standardizes the train, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    num_classes = None  # overwrite in subclass
    # self.dims is returned when you call datamodule.size()
    dims = ()  # overwrite in subclass

    def __init__(
        self,
        flatten_input: bool,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        data_dir: str = "data/",
    ):
        super().__init__()

        self.flatten_input = flatten_input
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        raise NotImplementedError

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if hasattr(self, "test_dataset"):
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
            )
        else:
            return None

    # End PyTorch Lightning interface


class GroupDataModule(BaseDataModule):
    def compute_weights(self, dataset):
        assert hasattr(dataset, "y_array")
        assert hasattr(dataset, "group_array")
        y_counter, g_counter = self.count(dataset)
        weights = torch.ones(len(dataset))
        group_weight_map = {g: 1 / count for g, count in g_counter.items()}
        for g, weight in group_weight_map.items():
            weights[dataset.group_array == g] *= weight
        return y_counter, g_counter, weights

    def count(self, dataset):
        y_counter = Counter(dataset.y_array.tolist())
        g_counter = Counter(dataset.group_array.tolist())
        return y_counter, g_counter


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
