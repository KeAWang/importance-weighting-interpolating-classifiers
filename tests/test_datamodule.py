import sys

sys.path.append("../")
from src.pl_datamodules.imbalanced_datamodule import ImbalancedDataModule, CIFAR10ImbalancedDataModule


datamodule = CIFAR10ImbalancedDataModule(desired_classes)
