import torch
from typing import Tuple
from math import prod
from .imbalanced_datamodule import BaseDataModule
from torchvision.transforms import transforms
from ..datasets.waterbirds_dataset import WaterbirdsDataset


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

        # Nothing to do here. We assume data is already downloaded
        return

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
