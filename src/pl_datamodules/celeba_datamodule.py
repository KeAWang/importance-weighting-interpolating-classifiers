import torch
import urllib.request
import tarfile

from typing import Tuple, List
from math import prod
from .base_datamodule import (
    GroupDataModule,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from torchvision.transforms import transforms
from ..datasets.celeba_dataset import CelebADataset
from ..datasets.utils import ReweightedDataset


class CelebADataModule(GroupDataModule):
    def __init__(
        self,
        resolution: Tuple[int, int],
        augment_data: bool,
        target_name: str,
        confounder_names: List[str],
        train_frac: int,
        train_weight_exponent: float = 1.0,
        **kwargs,
    ):
        resolution = tuple(resolution)
        super().__init__(**kwargs)
        self.dims = (3,) + resolution
        self.num_classes = 2
        self.augment_data = augment_data
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.train_frac = train_frac
        self.train_weight_exponent = train_weight_exponent

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

        dataset_splits = full_dataset.get_splits(
            ["train", "val", "test"], train_frac=self.train_frac, seed=0
        )
        train_dataset = dataset_splits["train"]
        val_dataset = dataset_splits["val"]
        test_dataset = dataset_splits["test"]

        test_dataset = subsample_groups(
            test_dataset, [180, 180, 180, 180], seed=1234
        )  # subsample so that test set is even across groups

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
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
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
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    return transforms_list


def subsample_groups(dataset, new_group_sizes: List[int], seed):
    from torch.utils.data import Subset
    import numpy as np

    group_array = dataset.group_array
    indices_kept = []
    rng = np.random.default_rng(seed=seed)
    assert len(new_group_sizes) == len(np.unique(group_array))
    for g in np.unique(group_array):
        g = int(g)
        g_indices = np.where(group_array == g)[0]
        g_indices_kept = rng.choice(g_indices, new_group_sizes[g], replace=False)
        indices_kept.append(g_indices_kept)

    indices = np.concatenate(indices_kept)
    new_dataset = Subset(dataset, indices)
    new_dataset.group_array = group_array[indices]
    new_dataset.y_array = dataset.y_array[indices]
    return new_dataset
