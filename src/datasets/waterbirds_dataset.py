import os
from typing import Tuple
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, Subset
from .utils import GroupedLabeledDatapoint


class ConfounderDataset(Dataset):
    """Modified from https://github.com/kohpangwei/group_DRO/blob/master/data/confounder_dataset.py

    root: the path to the directory containing the directory containing the images
    """

    def __init__(
        self, root: str, resolution: Tuple[int, int], augment_data: str,
    ):
        raise NotImplementedError

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert("RGB")
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict["train"] and self.train_transform:
            img = self.train_transform(img)
        elif (
            self.split_array[idx] in [self.split_dict["val"], self.split_dict["test"]]
            and self.eval_transform
        ):
            img = self.eval_transform(img)

        x = img

        return GroupedLabeledDatapoint(x=x, y=y, g=g)

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ("train", "val", "test"), split + " is not a valid split"
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            if train_frac < 1 and split == "train":
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
            subsets[split].group_array = self.group_array[indices]
            subsets[split].y_array = self.y_array[indices]

        return subsets

    def group_str(self, group_idx):
        # TODO: what is this function
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
        return group_name


class WaterbirdsDataset(ConfounderDataset):
    """ From https://github.com/kohpangwei/group_DRO/blob/master/data/cub_dataset.py
    Returns CUB dataset along with the transforms (which crop and center the original images).
    Note: metadata_df is one-indexed.
    """

    # Expect dataset to be in root_dir/base_folder
    base_folder = "waterbirds"

    def __init__(self, root, train_transform, eval_transform):
        self.root = root

        # The unziped tarball directory. The tarball can be found at https://worksheets.codalab.org/bundles/0xb922b6c2d39c48bab4516780e06d5649
        self.data_dir = os.path.join(self.root, self.base_folder)
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in metadata
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))

        # Get the y values
        # NOTE: list of 1s and 0s. 0 is landbird and 1 is waterbird
        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        # NOTE: list of 1s and 0s, representing the confounder
        # NOTE: I think 0 is land and 1 is water
        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df["place"].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        # group mappings:
        #   0: landbird, land
        #   1: landbird, water
        #   2: waterbird, land
        #   3: waterbird, water
        self.group_array = (
            self.y_array * (self.n_groups / 2) + self.confounder_array
        ).astype("int")

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        # Set transform
        self.train_transform = train_transform
        self.eval_transform = eval_transform
