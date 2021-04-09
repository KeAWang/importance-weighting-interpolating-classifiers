#!/bin/bash 

python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[16, 16]" datamodule.num_oversample_per_train_class="[16, 16]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[64, 64]" datamodule.num_oversample_per_train_class="[64, 64]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[256, 256]" datamodule.num_oversample_per_train_class="[256, 256]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[1024, 1024]" datamodule.num_oversample_per_train_class="[1024, 1024]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[4096, 4096]" datamodule.num_oversample_per_train_class="[4096, 4096]"
