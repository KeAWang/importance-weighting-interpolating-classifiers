from .base_datamodule import (
    GroupDataModule,
    CIFAR_DEFAULT_MEAN,
    CIFAR_DEFAULT_STD,
)
from math import prod
from torchvision.transforms import transforms
from ..datasets.imbalanced_cifar_datasets import ImbalancedCIFAR10Dataset
from ..datasets.utils import ReweightedDataset
import torch


class ImbalancedCIFAR10DataModule(GroupDataModule):
    # TODO: add option of resizing image
    dims = (3, 32, 32)
    num_classes = 10
    """We use the class label as the group label"""

    def __init__(
        self, imb_type: str, imb_factor: int, data_augmentation: bool = False, **kwargs
    ):
        super().__init__(**kwargs)

        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.data_augmentation = data_augmentation

        # transforms_list = [
        #    transforms.ToTensor(),
        #    transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
        # ]
        # if self.flatten_input:
        #    transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        #    self.dims = (prod(self.dims),)

        # if data_augmentation:
        #    train_transforms = [
        #        transforms.RandomCrop(32, padding=4),
        #        transforms.RandomHorizontalFlip(),
        #    ] + transforms_list
        # else:
        #    train_transforms = transforms_list
        # eval_transforms = transforms_list
        if data_augmentation:
            train_transforms = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
            ]
        else:
            train_transforms = [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
            ]
        eval_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
        ]

        self.train_transform = transforms.Compose(train_transforms)
        self.eval_transform = transforms.Compose(eval_transforms)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.prepare_base_train_dataset()
        self.prepare_base_val_dataset()

    def prepare_base_train_dataset(self):
        dataset = ImbalancedCIFAR10Dataset(
            root=self.data_dir,
            imb_type=self.imb_type,
            imb_factor=self.imb_factor,
            train=True,
            transform=self.train_transform,
            download=True,
        )
        return dataset

    def prepare_base_val_dataset(self):
        # Note: Using test set as validation set
        dataset = ImbalancedCIFAR10Dataset(
            root=self.data_dir,
            imb_type="none",
            train=False,
            transform=self.eval_transform,
            download=True,
        )
        return dataset

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        # get only datapoints that belong to classes
        base_train_dataset = self.prepare_base_train_dataset()
        self.train_y_counter, self.train_g_counter = self.count(base_train_dataset)
        print(f"Train class counts: {self.train_y_counter}")
        print(f"Train group counts: {self.train_g_counter}")
        self.train_dataset = base_train_dataset

        base_val_dataset = self.prepare_base_val_dataset()
        _, _, val_weights = self.compute_weights(base_val_dataset)
        self.val_dataset = ReweightedDataset(base_val_dataset, weights=val_weights)

        # TODO: don't validate on test set?


class ImbalancedCIFAR100DataModule(ImbalancedCIFAR10DataModule):
    num_classes = 100
