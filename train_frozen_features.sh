#!/bin/bash
# First train on imbalanced data for learning features
#python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[1024, 4096]" datamodule.num_oversample_per_train_class="[1024, 4096]" seed=0

# Now train linear classifiers by initializing with previous linear layer with undersampled data but different undersampling seeds
#/mnt/storage1/Documents/repos/importance-reweighing/logs/runs/2021-04-16/08-56-11
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[1024, 1024]" datamodule.num_oversample_per_train_class="[1024, 1024]" seed=1 load_wandb_run="['kealexanderwang', 'importance-reweighing', 'restful-monkey-84', 'last.ckpt']" model.freeze_features=True trainer.max_epochs=100 logger.extra_tags="['resampled-linear']"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[1024, 1024]" datamodule.num_oversample_per_train_class="[1024, 1024]" seed=2 load_wandb_run="['kealexanderwang', 'importance-reweighing', 'restful-monkey-84', 'last.ckpt']" model.freeze_features=True trainer.max_epochs=100 logger.extra_tags="['resampled-linear']"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[1024, 1024]" datamodule.num_oversample_per_train_class="[1024, 1024]" seed=3 load_wandb_run="['kealexanderwang', 'importance-reweighing', 'restful-monkey-84', 'last.ckpt']" model.freeze_features=True trainer.max_epochs=100 logger.extra_tags="['resampled-linear']"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[1024, 1024]" datamodule.num_oversample_per_train_class="[1024, 1024]" seed=4 load_wandb_run="['kealexanderwang', 'importance-reweighing', 'restful-monkey-84', 'last.ckpt']" model.freeze_features=True trainer.max_epochs=100 logger.extra_tags="['resampled-linear']"
python run.py +experiment=cifar10_binary datamodule.num_undersample_per_train_class="[1024, 1024]" datamodule.num_oversample_per_train_class="[1024, 1024]" seed=5 load_wandb_run="['kealexanderwang', 'importance-reweighing', 'restful-monkey-84', 'last.ckpt']" model.freeze_features=True trainer.max_epochs=100 logger.extra_tags="['resampled-linear']"
