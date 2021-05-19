from typing import Callable, Optional
from torch import Generator
from torch.utils.data import Dataset
import torchvision.datasets
import numpy as np

from .utils import GroupedLabeledDatapoint, UndersampledByGroupDataset


class ImbalancedCIFAR10Dataset(Dataset):
    num_classes = 10
    cifar_cls_name = "CIFAR10"

    def __init__(
        self,
        root: str,
        imb_type: str,
        imb_factor: float,
        generator: Optional[Generator] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        cifar_dataset = getattr(torchvision.datasets, self.cifar_cls_name)(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        labels = [y for x, y in cifar_dataset]
        new_group_sizes = type(self)._get_img_num_per_cls(
            cifar_dataset,
            num_classes=self.num_classes,
            imb_type=imb_type,
            imb_factor=imb_factor,
        )

        self.undersampled_cifar = UndersampledByGroupDataset(
            cifar_dataset,
            group_ids=labels,
            new_group_sizes=new_group_sizes,
            generator=generator,
        )
        self.y_array = np.array(labels)[self.undersampled_cifar.indices]
        self.group_array = self.y_array

    @staticmethod
    def _get_img_num_per_cls(cifar_dataset, num_classes, imb_type, imb_factor):
        """Modified from https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py"""
        img_max = len(cifar_dataset) / num_classes  # CIFAR datasets are balanced
        img_num_per_cls = []
        if imb_type == "exp":
            for cls_idx in range(num_classes):
                num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(num_classes // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        elif imb_type == "none":
            img_num_per_cls.extend([int(img_max)] * num_classes)
        else:
            raise ValueError(f"Imbalance type imb_type={imb_type} not supported")
        img_num_per_cls = dict(enumerate(img_num_per_cls))
        return img_num_per_cls

    def __getitem__(self, idxs):
        x, y = self.undersampled_cifar[idxs]
        return GroupedLabeledDatapoint(x=x, y=y, g=y)

    def __len__(self):
        return len(self.undersampled_cifar)


class ImbalancedCIFAR100Dataset(ImbalancedCIFAR10Dataset):
    num_classes = 100
    cifar_cls_name = "CIFAR100"
