#!/bin/bash 
# ConvNet
python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 4096]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 3072]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 2048]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 1024]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1

# MLP
python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 4096]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=mlp_net

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 3072]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=mlp_net

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 2048]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=mlp_net

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 1024]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=mlp_net

# Logistic regression
python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 4096]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=linear_net

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 3072]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=linear_net

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 2048]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=linear_net

python run.py +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 1024]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
optimizer.lr=0.01 trainer.max_epochs=100  \
callbacks.model_checkpoint.period=5 \
callbacks.model_checkpoint.save_top_k=-1 \
datamodule.flatten_input=True \
architecture=linear_net
