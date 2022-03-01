import torch
from typing import List, Sequence, Optional
from math import prod
from pathlib import Path

from ..datasets.imbalanced_cifar10_dataset import ImbalancedCIFAR10
from ..datasets.imbalanced_moons_dataset import ImbalancedMoonsDataset
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImbalancedCIFAR10DataModule(BaseDataModule):
    dims = (3, 32, 32)

    def __init__(
        self,
        desired_classes: List[int] = None,
        num_undersample_per_train_class: List[int] = None,
        num_oversample_per_train_class: List[int] = None,
        num_undersample_per_test_class: List[int] = None,
        num_oversample_per_test_class: List[int] = None,
        **kwargs,
    ):
        """
        desired_classes: None indicates keep all classes
        num_*sample_per_*_class: None indicates don't *sample
        """
        super().__init__(**kwargs)

        desired_classes = (
            list(set(desired_classes))
            if desired_classes is not None
            else list(range(10))
        )

        if (
            num_undersample_per_train_class is None
            and num_oversample_per_train_class is None
        ):
            num_undersample_per_train_class = [5000] * len(desired_classes)
            num_oversample_per_train_class = [5000] * len(desired_classes)
        if (
            num_undersample_per_train_class is not None
            and num_oversample_per_train_class is None
        ):
            num_oversample_per_train_class = list(num_undersample_per_train_class)

        if (
            num_undersample_per_test_class is None
            and num_oversample_per_test_class is None
        ):
            num_undersample_per_test_class = [1000] * len(desired_classes)
            num_oversample_per_test_class = [1000] * len(desired_classes)

        if (
            num_undersample_per_test_class is not None
            and num_oversample_per_test_class is None
        ):
            num_oversample_per_test_class = list(num_undersample_per_test_class)

        self.num_classes = len(desired_classes)
        self.desired_classes = desired_classes
        self.num_undersample_per_train_class = num_undersample_per_train_class
        self.num_oversample_per_train_class = num_oversample_per_train_class
        self.num_undersample_per_test_class = num_undersample_per_test_class
        self.num_oversample_per_test_class = num_oversample_per_test_class

        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
            ),  # see https://github.com/kuangliu/pytorch-cifar/issues/19 for example
        ]
        if self.flatten_input:
            transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
            self.dims = (prod(self.dims),)

        self.transforms = transforms.Compose(transforms_list)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.prepare_train_dataset()
        self.prepare_val_dataset()

    def prepare_train_dataset(self):
        dataset = ImbalancedCIFAR10(
            self.data_dir,
            desired_classes=self.desired_classes,
            num_undersample_per_class=self.num_undersample_per_train_class,
            num_oversample_per_class=self.num_oversample_per_train_class,
            train=True,
            download=True,
            transform=self.transforms,
        )
        return dataset

    def prepare_val_dataset(self):
        # Note: Using test set as validation set
        dataset = ImbalancedCIFAR10(
            self.data_dir,
            desired_classes=self.desired_classes,
            num_undersample_per_class=self.num_undersample_per_test_class,
            num_oversample_per_class=self.num_oversample_per_test_class,
            train=False,
            download=True,
            transform=self.transforms,
        )
        return dataset

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        # get only datapoints that belong to classes
        self.train_dataset = self.prepare_train_dataset()
        self.val_dataset = self.prepare_val_dataset()
        # TODO: don't validate on test set?


class ImbalancedMoonsDataModule(BaseDataModule):
    dims = (2,)

    def __init__(
        self,
        num_undersample_per_train_class: List[int] = None,
        num_oversample_per_train_class: List[int] = None,
        num_undersample_per_test_class: List[int] = None,
        num_oversample_per_test_class: List[int] = None,
        **kwargs,
    ):
        """
        desired_classes: None indicates keep all classes
        num_*sample_per_*_class: None indicates don't *sample
        """
        super().__init__(**kwargs)

        if (
            num_undersample_per_train_class is None
            and num_oversample_per_train_class is None
        ):
            num_undersample_per_train_class = [512, 512]
            num_oversample_per_train_class = [512, 512]
        if (
            num_undersample_per_train_class is not None
            and num_oversample_per_train_class is None
        ):
            num_oversample_per_train_class = list(num_undersample_per_train_class)

        if (
            num_undersample_per_test_class is None
            and num_oversample_per_test_class is None
        ):
            num_undersample_per_test_class = [512, 512]
            num_oversample_per_test_class = [512, 512]

        if (
            num_undersample_per_test_class is not None
            and num_oversample_per_test_class is None
        ):
            num_oversample_per_test_class = list(num_undersample_per_test_class)

        self.num_classes = 2
        self.num_undersample_per_train_class = num_undersample_per_train_class
        self.num_oversample_per_train_class = num_oversample_per_train_class
        self.num_undersample_per_test_class = num_undersample_per_test_class
        self.num_oversample_per_test_class = num_oversample_per_test_class

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.prepare_train_dataset()
        self.prepare_val_dataset()

    def prepare_train_dataset(self):
        dataset = ImbalancedMoonsDataset(
            num_undersample_per_class=self.num_undersample_per_train_class,
            num_oversample_per_class=self.num_oversample_per_train_class,
            num_samples=(512, 512),
        )
        return dataset

    def prepare_val_dataset(self):
        dataset = ImbalancedMoonsDataset(
            num_undersample_per_class=self.num_undersample_per_train_class,
            num_oversample_per_class=self.num_oversample_per_train_class,
            num_samples=(512, 512),
        )
        return dataset

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        self.train_dataset = self.prepare_train_dataset()
        self.val_dataset = self.prepare_val_dataset()
