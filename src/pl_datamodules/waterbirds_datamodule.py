import torch
import urllib.request
import tarfile

from typing import Tuple, Callable, Optional
from math import prod
from .base_datamodule import BaseDataModule
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from ..datasets.waterbirds_dataset import WaterbirdsDataset
from ..datasets.utils import (
    ResampledDataset,
    UndersampledDataset,
    ReweightedDataset,
    undersampling_schedule,
)
from collections import Counter


class WaterbirdsDataModule(BaseDataModule):
    def __init__(
        self, resolution, augment_data, **kwargs,
    ):
        resolution = tuple(resolution)
        super().__init__(**kwargs)
        self.dims = (3,) + resolution
        self.num_classes = 2
        self.augment_data = augment_data

        train_transforms_list = get_transforms_list(
            resolution, train=True, augment_data=augment_data
        )
        eval_transforms_list = get_transforms_list(
            resolution, train=False, augment_data=False
        )

        if self.flatten_input:
            train_transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
            eval_transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

            self.dims = (prod(self.dims),)

        self.train_transform = transforms.Compose(train_transforms_list)
        self.eval_transform = transforms.Compose(eval_transforms_list)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        dataset_dir = self.data_dir / WaterbirdsDataset.base_folder
        if not dataset_dir.exists():
            print("Downloading Waterbirds dataset...")
            url = "https://worksheets.codalab.org/rest/bundles/0xb922b6c2d39c48bab4516780e06d5649/contents/blob/"
            file_path = self.data_dir / "waterbirds.tar.gz"
            urllib.request.urlretrieve(url, file_path)
            with tarfile.open(file_path) as file:
                file.extractall(dataset_dir)

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        full_dataset = WaterbirdsDataset(
            self.data_dir,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        dataset_splits = full_dataset.get_splits(["train", "val", "test"])
        self.train_dataset = dataset_splits["train"]
        self.val_dataset = dataset_splits["val"]
        self.test_dataset = dataset_splits["test"]

        self.train_y_counter, self.train_g_counter = self.count(self.train_dataset)
        print(f"Train class counts: {self.train_y_counter}")
        print(f"Train group counts: {self.train_g_counter}")
        self.val_y_counter, self.val_g_counter = self.count(self.val_dataset)
        print(f"Val class counts: {self.val_y_counter}")
        print(f"Val group counts: {self.val_g_counter}")
        self.test_y_counter, self.test_g_counter = self.count(self.test_dataset)
        print(f"Test class counts: {self.test_y_counter}")
        print(f"Test group counts: {self.test_g_counter}")

    def count(self, dataset):
        y_counter = Counter(dataset.y_array.tolist())
        g_counter = Counter(dataset.group_array.tolist())
        return y_counter, g_counter


class ReweightedWaterbirdsDataModule(WaterbirdsDataModule):
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        # Resample train dataset
        train_dataset = self.train_dataset
        train_weights = torch.ones(len(train_dataset))
        group_weight_map = {g: 1 / count for g, count in self.train_g_counter.items()}
        for g, weight in group_weight_map.items():
            train_weights[train_dataset.group_array == g] *= weight
        reweighted_train_dataset = ReweightedDataset(
            train_dataset, weights=train_weights
        )
        self.train_dataset = reweighted_train_dataset

        # Keep val and test dataset the same as before


class ResampledWaterbirdsDataModule(WaterbirdsDataModule):
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        # Resample train dataset
        train_dataset = self.train_dataset
        train_weights = torch.ones(len(train_dataset))
        group_weight_map = {g: 1 / count for g, count in self.train_g_counter.items()}
        for g, weight in group_weight_map.items():
            train_weights[train_dataset.group_array == g] *= weight
        resampled_train_dataset = ResampledDataset(
            train_dataset, weights=train_weights, new_size=len(train_dataset)
        )
        self.train_dataset = resampled_train_dataset

        # Keep val and test dataset the same as before


class UndersampledWaterbirdsDataModule(WaterbirdsDataModule):
    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        # Resample train dataset
        train_dataset = self.train_dataset
        train_weights = torch.ones(len(train_dataset))
        group_weight_map = {g: 1 / count for g, count in self.train_g_counter.items()}
        for g, weight in group_weight_map.items():
            train_weights[train_dataset.group_array == g] *= weight
        undersampled_train_dataset = UndersampledDataset(
            train_dataset, weights=train_weights
        )

        new_indices = undersampled_train_dataset.indices
        new_group_array = train_dataset.group_array[new_indices]
        new_y_array = train_dataset.y_array[new_indices]
        undersampled_train_dataset.group_array = new_group_array
        undersampled_train_dataset.y_array = new_y_array

        self.train_dataset = undersampled_train_dataset
        self.train_y_counter, self.train_g_counter = self.count(self.train_dataset)
        print(f"Dataset classes were undersampled to {self.train_y_counter}")
        print(f"Dataset groups were undersampled to {self.train_g_counter}")

        # Keep val and test dataset the same as before


def get_transforms_list(resolution: Tuple[int, int], train: bool, augment_data: bool):
    scale = 256.0 / 224.0
    assert resolution is not None

    # Only train dataset has the option of being data-augmented
    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transforms_list = [
            transforms.Resize(
                (int(resolution[0] * scale), int(resolution[1] * scale),)
            ),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        transforms_list = [
            transforms.RandomResizedCrop(
                resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    return transforms_list


class AnnealedWaterbirdsDataModule(WaterbirdsDataModule):
    def __init__(
        self, annealing_fn: Optional[Callable], num_epochs: int, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if annealing_fn is None:

            def annealing_fn(weights, t, T):
                beta = 1 - (t / T)
                return weights ** beta

        self.current_epoch = 0
        self.annealing_fn = annealing_fn
        self.num_epochs = num_epochs

    def setup(self, stage=None):
        super().setup(stage=stage)
        # Compute weights
        train_dataset = self.train_dataset
        train_weights = torch.ones(len(train_dataset))
        group_weight_map = {g: 1 / count for g, count in self.train_g_counter.items()}
        for g, weight in group_weight_map.items():
            train_weights[train_dataset.group_array == g] *= weight

        self.annealed_train_datasets = list(
            ReweightedDataset(Subset(train_dataset, idxs), ws)
            for idxs, ws in undersampling_schedule(
                train_weights, self.num_epochs, self.annealing_fn
            )
        )

    def train_dataloader(self):
        dataset = self.annealed_train_datasets[self.current_epoch]
        # NOTE: Must use with Trainer.reload_dataloaders_every_epoch = True
        self.current_epoch += 1
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )
