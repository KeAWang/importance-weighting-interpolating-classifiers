from .base_datamodule import (
    GroupDataModule,
    CIFAR_DEFAULT_MEAN,
    CIFAR_DEFAULT_STD,
)
from math import prod
from typing import List, Optional
from torchvision.transforms import transforms
from ..datasets.imbalanced_cifar_datasets import ImbalancedCIFAR10Dataset
from ..datasets.utils import ReweightedDataset, split_dataset, SubsetOfGroupedDataset
import torch


class ImbalancedCIFAR10DataModule(GroupDataModule):
    # TODO: add option of resizing image
    dims = (3, 32, 32)
    num_classes = 10
    """We use the class label as the group label"""

    def __init__(
        self,
        imb_type: str,
        imb_factor: int,
        data_augmentation: bool = False,
        class_subset: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.data_augmentation = data_augmentation
        self.class_subset = (
            list(set(class_subset)) if class_subset is not None else None
        )
        if class_subset is not None:
            self.num_classes = len(self.class_subset)
            self.old_to_new_class = {y: i for i, y in enumerate(self.class_subset)}
            self.new_to_old_class = {i: y for i, y in enumerate(self.class_subset)}

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
        self.prepare_base_test_dataset()

    def prepare_base_train_dataset(self):
        dataset = ImbalancedCIFAR10Dataset(
            root=self.data_dir,
            imb_type=self.imb_type,
            imb_factor=self.imb_factor,
            train=True,
            transform=self.train_transform,
            download=True,
        )
        if self.class_subset is not None:
            chosen = [
                i for i, y in enumerate(dataset.y_array) if int(y) in self.class_subset
            ]
            dataset = SubsetOfGroupedDataset(
                dataset,
                chosen,
                old_to_new_class=self.old_to_new_class,
                old_to_new_group=self.old_to_new_class,
            )
        return dataset

    def prepare_base_test_dataset(self):
        # Note: Using test set as validation set
        dataset = ImbalancedCIFAR10Dataset(
            root=self.data_dir,
            imb_type="none",
            train=False,
            transform=self.eval_transform,
            download=True,
        )
        if self.class_subset is not None:
            chosen = [
                i for i, y in enumerate(dataset.y_array) if int(y) in self.class_subset
            ]
            dataset = SubsetOfGroupedDataset(
                dataset,
                chosen,
                old_to_new_class=self.old_to_new_class,
                old_to_new_group=self.old_to_new_class,
            )
        return dataset

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        base_train_dataset = self.prepare_base_train_dataset()
        num_train_full = len(base_train_dataset)
        num_train = int(0.8 * num_train_full)  # 80/20 split
        # WARNING: val dataset will have the same transforms as the train dataset!
        train_dataset, val_dataset = split_dataset(
            base_train_dataset, [num_train], shuffle=True, seed=0
        )  # always use seed 0 for split

        test_dataset = self.prepare_base_test_dataset()

        self.train_y_counter, self.train_g_counter = self.count(train_dataset)
        print(f"Train class counts: {self.train_y_counter}")
        print(f"Train group counts: {self.train_g_counter}")
        self.train_dataset = train_dataset

        self.val_y_counter, self.val_g_counter = self.count(val_dataset)
        print(f"Val class counts: {self.val_y_counter}")
        print(f"Val group counts: {self.val_g_counter}")
        _, _, val_weights = self.compute_weights(val_dataset)
        val_dataset = ReweightedDataset(val_dataset, weights=val_weights)
        self.val_dataset = val_dataset

        self.test_y_counter, self.test_g_counter = self.count(test_dataset)
        print(f"Test class counts: {self.test_y_counter}")
        print(f"Test group counts: {self.test_g_counter}")
        _, _, test_weights = self.compute_weights(test_dataset)
        test_dataset = ReweightedDataset(test_dataset, weights=test_weights)
        self.test_dataset = test_dataset


class ImbalancedCIFAR100DataModule(ImbalancedCIFAR10DataModule):
    num_classes = 100
