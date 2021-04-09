#!/bin/bash 

python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[16, 4096]" datamodule.num_oversample_per_train_class="[16, 4096]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[64, 4096]" datamodule.num_oversample_per_train_class="[64, 4096]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[256, 4096]" datamodule.num_oversample_per_train_class="[256, 4096]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[1024, 4096]" datamodule.num_oversample_per_train_class="[1024, 4096]"
python run.py +experiment=cifar10_binary_reweigh datamodule.num_undersample_per_train_class="[4096, 4096]" datamodule.num_oversample_per_train_class="[4096, 4096]"
