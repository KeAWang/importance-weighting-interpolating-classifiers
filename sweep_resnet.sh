#!/bin/bash 
python run.py --multirun +experiment=cifar10_binary +logger.extra_tags="[early-stopping]" seed=0 \
datamodule.num_undersample_per_train_class="[1024, 4096]" \
datamodule.num_oversample_per_train_class="[4096, 4096]" \
architecture=res_net architecture.k=64,128,256 \
optimizer.lr=0.01 trainer.max_epochs=100  \
#callbacks.model_checkpoint.period=5 \
#callbacks.model_checkpoint.save_top_k=-1
