from .base_datamodule import GroupDataModule
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

    def __init__(self, imb_type, imb_factor, **kwargs):
        super().__init__(**kwargs)

        self.imb_type = imb_type
        self.imb_factor = imb_factor

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
        self.prepare_base_train_dataset()
        self.prepare_base_val_dataset()

    def prepare_base_train_dataset(self):
        dataset = ImbalancedCIFAR10Dataset(
            root=self.data_dir,
            imb_type=self.imb_type,
            imb_factor=self.imb_factor,
            train=True,
            transform=self.transforms,
            download=True,
        )
        return dataset

    def prepare_base_val_dataset(self):
        # Note: Using test set as validation set
        dataset = ImbalancedCIFAR10Dataset(
            root=self.data_dir,
            desired_classes=self.desired_classes,
            num_undersample_per_class=self.num_undersample_per_test_class,
            num_oversample_per_class=self.num_oversample_per_test_class,
            train=False,
            transform=self.transforms,
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
