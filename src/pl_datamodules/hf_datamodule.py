import torch

from typing import Optional
from .base_datamodule import GroupDataModule
from torchvision.transforms import transforms
from datasets import load_dataset
from ..datasets.utils import ReweightedDataset, UndersampledByGroupDataset
from ..datasets.mnli_dataset import MNLIDataset
from transformers.tokenization_utils import PreTrainedTokenizer


class MNLIDataModule(GroupDataModule):
    dataset_name = "multi_nli"
    num_classes = 3  # entailment (0), neutral (1), contradiction (2)
    dims = None  # TODO

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        train_frac: float,
        new_group_sizes: Optional = None,
        new_group_fracs: Optional = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.train_frac = train_frac
        self.new_group_sizes = new_group_sizes
        self.new_group_fracs = new_group_fracs

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        _ = load_dataset(self.dataset_name, cache_dir=self.data_dir)

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        full_dataset = load_dataset(self.dataset_name, cache_dir=self.data_dir)

        train_transform = init_transform(self.tokenizer)
        val_transform = init_transform(self.tokenizer)

        train_dataset = MNLIDataset(
            full_dataset["train"].filter(lambda example: example["label"] != -1),
            input_transform=train_transform,
            frac=self.train_frac,
        )
        if self.new_group_sizes is not None or self.new_group_fracs is not None:
            new_train_dataset = UndersampledByGroupDataset(
                train_dataset,
                train_dataset.group_array,
                self.new_group_sizes,
                self.new_group_fracs,
            )
            new_train_dataset.y_array = train_dataset.y_array[new_train_dataset.indices]
            new_train_dataset.group_array = train_dataset.group_array[
                new_train_dataset.indices
            ]
            train_dataset = new_train_dataset

        val_dataset = MNLIDataset(
            full_dataset["validation_matched"].filter(
                lambda example: example["label"] != -1
            ),
            input_transform=val_transform,
        )

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

        val_dataset = ReweightedDataset(val_dataset, weights=val_weights)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


def init_transform(tokenizer):
    def transform_inference(premise__hypothesis):
        premise, hypothesis = premise__hypothesis
        encodings = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        if tokenizer.name_or_path == "bert-base-uncased":
            x = torch.stack(
                (
                    encodings["input_ids"],
                    encodings["attention_mask"],
                    encodings["token_type_ids"],
                ),
                dim=2,
            )
        elif tokenizer.name_or_path == "distilbert-base-uncased":
            x = torch.stack(
                (encodings["input_ids"], encodings["attention_mask"]), dim=2
            )
        else:
            raise RuntimeError
        x = torch.squeeze(
            x, dim=0
        )  # First shape dim is always 1 since we're not in batch mode
        return x

    transform = transforms.Lambda(lambda x: transform_inference(x))
    return transform
