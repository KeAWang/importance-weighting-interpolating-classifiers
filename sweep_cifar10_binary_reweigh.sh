#!/bin/bash 
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[4096, 4096]" datamodule.num_oversample_per_train_class="[4096, 4096]" trainer.max_epochs=500 model.class_weights="[1, 256]"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[4096, 4096]" datamodule.num_oversample_per_train_class="[4096, 4096]" trainer.max_epochs=500 model.class_weights="[1, 64]"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[4096, 4096]" datamodule.num_oversample_per_train_class="[4096, 4096]" trainer.max_epochs=500 model.class_weights="[1, 16]"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[4096, 4096]" datamodule.num_oversample_per_train_class="[4096, 4096]" trainer.max_epochs=500 model.class_weights="[1, 4]"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[4096, 4096]"gdatamodule.num_oversample_per_train_class="[4096, 4096]" trainer.max_epochs=500 model.class_weights="[1, 1]"
