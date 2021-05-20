import torch
import urllib.request
import tarfile

from typing import Tuple
from math import prod
from .base_datamodule import GroupDataModule
from torchvision.transforms import transforms
from ..datasets.waterbirds_dataset import WaterbirdsDataset
from ..datasets.utils import ReweightedDataset
from .utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class WaterbirdsDataModule(GroupDataModule):
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
        file_path = self.data_dir / "waterbirds.tar.gz"
        if not file_path.exists():
            print("Downloading Waterbirds dataset...")
            url = "https://worksheets.codalab.org/rest/bundles/0xb922b6c2d39c48bab4516780e06d5649/contents/blob/"
            urllib.request.urlretrieve(url, file_path)
        elif not dataset_dir.exists():
            print("Extracting waterbirds.tar.gz")
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
        train_dataset = dataset_splits["train"]
        val_dataset = dataset_splits["val"]
        test_dataset = dataset_splits["test"]

        self.train_y_counter, self.train_g_counter, _ = self.compute_weights(
            train_dataset
        )
        print(f"Train class counts: {self.train_y_counter}")
        print(f"Train group counts: {self.train_g_counter}")
        self.val_y_counter, self.val_g_counter, val_weights = self.compute_weights(
            val_dataset
        )
        print(f"Val class counts: {self.val_y_counter}")
        print(f"Val group counts: {self.val_g_counter}")
        self.test_y_counter, self.test_g_counter, test_weights = self.compute_weights(
            test_dataset
        )
        print(f"Test class counts: {self.test_y_counter}")
        print(f"Test group counts: {self.test_g_counter}")

        val_dataset = ReweightedDataset(val_dataset, weights=val_weights)
        test_dataset = ReweightedDataset(test_dataset, weights=test_weights)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


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
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
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
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    return transforms_list
