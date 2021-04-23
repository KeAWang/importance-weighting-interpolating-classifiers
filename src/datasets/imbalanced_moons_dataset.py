from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset
from typing import Union, Tuple, Optional, List
from .mixins import ImbalancedDatasetMixin
from numpy.random import RandomState
import torch


class ImbalancedMoonsDataset(TensorDataset, ImbalancedDatasetMixin):
    def __init__(
        self,
        num_undersample_per_class: List[int] = None,
        num_oversample_per_class: List[int] = None,
        num_samples: Union[Tuple[int, int], int] = (512, 512),
        shuffle: bool = True,
        noise: Optional[float] = None,
        random_state: Optional[Union[int, RandomState]] = None,
    ):
        inputs, targets = make_moons(
            n_samples=num_samples,
            shuffle=shuffle,
            noise=noise,
            random_state=random_state,
        )

        inputs, targets = self.resample_dataset(
            inputs,
            targets,
            desired_classes=[0, 1],
            num_undersample_per_class=num_undersample_per_class,
            num_oversample_per_class=num_oversample_per_class,
        )
        self.class_weights = self.compute_weights(targets)
        inputs = torch.as_tensor(inputs, dtype=torch.get_default_dtype())
        targets = torch.as_tensor(targets, dtype=torch.long)

        super().__init__(inputs, targets)
