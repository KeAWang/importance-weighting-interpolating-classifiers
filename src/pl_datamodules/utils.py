# import these datamodule for getattr later
from .waterbirds_datamodule import WaterbirdsDataModule
from .celeba_datamodule import CelebADataModule
from .imbalanced_cifar_datamodule import (
    ImbalancedCIFAR10DataModule,
    ImbalancedCIFAR100DataModule,
)
from .wilds_datamodule import WILDSDataModule

from .base_datamodule import GroupDataModule
from torch.utils.data import DataLoader, Subset
from ..datasets.utils import (
    ResampledDataset,
    UndersampledDataset,
    ReweightedDataset,
    undersampling_schedule,
)
from typing import Optional
import sys

this_module = sys.modules[__name__]


def make_datamodule(
    group_datamodule_cls_name, wrapper_type: Optional[str], **module_kwargs
):
    group_datamodule_cls = getattr(this_module, group_datamodule_cls_name)
    assert issubclass(group_datamodule_cls, GroupDataModule)
    if wrapper_type == "reweighted":

        class ReweightedDataModule(group_datamodule_cls):
            def setup(self, *args, **kwargs):
                super().setup(*args, **kwargs)
                # Reweigh train dataset
                train_dataset = self.train_dataset
                _, _, train_weights = self.compute_weights(train_dataset)
                reweighted_train_dataset = ReweightedDataset(
                    train_dataset, weights=train_weights
                )
                self.train_dataset = reweighted_train_dataset
                # Keep val and test dataset the same as before

        return ReweightedDataModule(**module_kwargs)

    elif wrapper_type == "resampled":

        class ResampledDataModule(group_datamodule_cls):
            def setup(self, *args, **kwargs):
                super().setup(*args, **kwargs)
                # Reweigh train dataset
                train_dataset = self.train_dataset
                _, _, train_weights = self.compute_weights(train_dataset)
                resampled_train_dataset = ResampledDataset(
                    train_dataset, weights=train_weights, new_size=len(train_dataset)
                )
                self.train_dataset = resampled_train_dataset
                # Keep val and test dataset the same as before

        return ResampledDataModule(**module_kwargs)

    elif wrapper_type == "undersampled":

        class UndersampledDataModule(group_datamodule_cls):
            def setup(self, *args, **kwargs):
                super().setup(*args, **kwargs)
                # Undersample train dataset
                train_dataset = self.train_dataset
                _, _, train_weights = self.compute_weights(train_dataset)
                undersampled_train_dataset = UndersampledDataset(
                    train_dataset, weights=train_weights
                )

                new_indices = undersampled_train_dataset.indices
                new_group_array = train_dataset.group_array[new_indices]
                new_y_array = train_dataset.y_array[new_indices]
                undersampled_train_dataset.group_array = new_group_array
                undersampled_train_dataset.y_array = new_y_array

                self.train_dataset = undersampled_train_dataset
                self.train_y_counter, self.train_g_counter = self.count(
                    self.train_dataset
                )
                print(f"Dataset classes were undersampled to {self.train_y_counter}")
                print(f"Dataset groups were undersampled to {self.train_g_counter}")

                # Keep val and test dataset the same as before

        return UndersampledDataModule(**module_kwargs)

    elif wrapper_type == "annealed":

        class AnnealedDataModule(group_datamodule_cls):
            def __init__(
                self,
                annealing_fn: str,
                num_epochs: int,
                annealing_params: list,
                *args,
                **kwargs,
            ):
                super().__init__(*args, **kwargs)

                self.current_epoch = 0
                self.annealing_fn = annealing_fn
                self.annealing_params = tuple(annealing_params)
                self.num_epochs = num_epochs

            def setup(self, stage=None):
                super().setup(stage=stage)
                # Compute weights
                train_dataset = self.train_dataset
                _, _, train_weights = self.compute_weights(train_dataset)

                self.annealed_train_datasets = list(
                    ReweightedDataset(Subset(train_dataset, idxs), ws)
                    for idxs, ws in undersampling_schedule(
                        train_weights,
                        self.num_epochs,
                        self.annealing_fn,
                        self.annealing_params,
                    )
                )

            def train_dataloader(self):
                dataset = self.annealed_train_datasets[self.current_epoch]
                # NOTE: Must use with Trainer.reload_dataloaders_every_epoch = True
                self.current_epoch += 1
                return DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=True,
                )

        return AnnealedDataModule(**module_kwargs)
    elif wrapper_type is None:
        return group_datamodule_cls(**module_kwargs)
