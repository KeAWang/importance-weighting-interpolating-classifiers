import torch
import urllib.request
import tarfile

from typing import Tuple, List
from math import prod
from .base_datamodule import BaseDataModule
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from ..datasets.celeba_dataset import CelebADataset
from ..datasets.utils import (
    ResampledDataset,
    UndersampledDataset,
    ReweightedDataset,
    undersampling_schedule,
)
from collections import Counter


class CelebADataModule(BaseDataModule):
    def __init__(
        self,
        resolution: Tuple[int, int],
        augment_data: bool,
        target_name: str,
        confounder_names: List[str],
        **kwargs,
    ):
        resolution = tuple(resolution)
        super().__init__(**kwargs)
        self.dims = (3,) + resolution
        self.num_classes = 2
        self.augment_data = augment_data
        self.target_name = target_name
        self.confounder_names = confounder_names

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

        dataset_dir = self.data_dir / CelebADataset.base_folder
        file_path = self.data_dir / "celeba.tar.gz"
        if not file_path.exists():
            print("Downloading CelebA dataset...")
            url = "https://worksheets.codalab.org/rest/bundles/0x886412315184400c9983b32846e91ab1/contents/blob/"
            urllib.request.urlretrieve(url, file_path)
        elif not dataset_dir.exists():
            print("Extracting celeba.tar.gz")
            with tarfile.open(file_path) as file:
                file.extractall(dataset_dir)

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        full_dataset = CelebADataset(
            self.data_dir,
            target_name=self.target_name,
            confounder_names=self.confounder_names,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        dataset_splits = full_dataset.get_splits(["train", "val", "test"])
        train_dataset = dataset_splits["train"]
        val_dataset = dataset_splits["val"]
        test_dataset = dataset_splits["test"]

        self.train_y_counter, self.train_g_counter = self.count(train_dataset)
        print(f"Train class counts: {self.train_y_counter}")
        print(f"Train group counts: {self.train_g_counter}")
        self.val_y_counter, self.val_g_counter = self.count(val_dataset)
        print(f"Val class counts: {self.val_y_counter}")
        print(f"Val group counts: {self.val_g_counter}")
        self.test_y_counter, self.test_g_counter = self.count(test_dataset)
        print(f"Test class counts: {self.test_y_counter}")
        print(f"Test group counts: {self.test_g_counter}")

        val_weights = torch.ones(len(val_dataset))
        group_weight_map = {g: 1 / count for g, count in self.val_g_counter.items()}
        for g, weight in group_weight_map.items():
            val_weights[val_dataset.group_array == g] *= weight
        val_dataset = ReweightedDataset(val_dataset, weights=val_weights)

        test_weights = torch.ones(len(test_dataset))
        group_weight_map = {g: 1 / count for g, count in self.test_g_counter.items()}
        for g, weight in group_weight_map.items():
            test_weights[test_dataset.group_array == g] *= weight
        test_dataset = ReweightedDataset(test_dataset, weights=test_weights)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def count(self, dataset):
        y_counter = Counter(dataset.y_array.tolist())
        g_counter = Counter(dataset.group_array.tolist())
        return y_counter, g_counter


class ReweightedCelebADataModule(CelebADataModule):
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


class ResampledCelebADataModule(CelebADataModule):
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


class UndersampledCelebADataModule(CelebADataModule):
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
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    assert resolution is not None

    # Only train dataset has the option of being data-augmented
    if (not train) or (not augment_data):
        transforms_list = [
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
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


class AnnealedCelebADataModule(CelebADataModule):
    def __init__(
        self,
        annealing_fn: str,
        num_epochs: int,
        annealing_params: list,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.current_epoch = 0
        self.annealing_fn = annealing_fn
        self.annealing_params = tuple(annealing_params)
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
                train_weights, self.num_epochs, self.annealing_fn, self.annealing_params
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
