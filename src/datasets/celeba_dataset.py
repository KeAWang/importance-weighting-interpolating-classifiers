import os
import numpy as np
import pandas as pd
from .waterbirds_dataset import ConfounderDataset


class CelebADataset(ConfounderDataset):
    """ From https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
    """

    # Expect dataset to be in root_dir/base_folder
    base_folder = "celeba"

    def __init__(
        self, root, target_name, confounder_names, train_transform, eval_transform
    ):
        self.root = root
        self.target_name = target_name
        self.confounder_names = confounder_names
        # The unziped tarball directory. The tarball can be found at https://worksheets.codalab.org/rest/bundles/0x886412315184400c9983b32846e91ab1/contents/blob/
        self.base_dir = os.path.join(self.root, self.base_folder)
        if not os.path.exists(self.base_dir):
            raise ValueError(
                f"{self.base_dir} does not exist yet. Please generate the dataset first."
            )

        # Read in attributes
        self.attrs_df = pd.read_csv(os.path.join(self.base_dir, "list_attr_celeba.csv"))

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.base_dir, "img_align_celeba")
        self.filename_array = self.attrs_df["image_id"].values
        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self.target_name)
        # NOTE: list of 1s and 0s. 0 is TODO and 1 is TODO
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        # NOTE: list of 1s and 0s, representing the confounder
        # NOTE: I think 0 is TODO and 1 is TODO
        # group mappings:
        #   0: TODO
        #   1: TODO
        #   2: TODO
        #   3: TODO
        self.group_array = (
            self.y_array * (self.n_groups / 2) + self.confounder_array
        ).astype("int")

        # Read in train/val/test splits
        self.split_df = pd.read_csv(
            os.path.join(self.base_dir, "list_eval_partition.csv")
        )
        self.split_array = self.split_df["partition"].values
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        # Set transform
        self.train_transform = train_transform
        self.eval_transform = eval_transform

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)
