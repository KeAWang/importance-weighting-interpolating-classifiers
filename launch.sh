#!/bin/bash
# See https://nlp.stanford.edu/local/cluster-info/index.html for usage
#mynlprun="nlprun -x jagupard[4-8],jagupard[19-20],jagupard[26-29]"
mynlprun="nlprun3 -x jagupard[11],jagupard[15],jagupard[18],jagupard[25-29]"

#for experiment in \
#    waterbirds_reweighted \
#    waterbirds_erm \
#    waterbirds_annealing \
#    waterbirds_resampled \
#    waterbirds_undersampled \
#
#do
#    for seed in 0
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done

#num_epochs=400
#experiment=waterbirds_annealing
#seed=0
#
#${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} optimizer.lr=0.00001 optimizer.weight_decay=1"\
#    -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"

#${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} optimizer.weight_decay=0.1"\
#    -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} datamodule.annealing_fn=step"\
#    -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"

#experiment=waterbirds_reference_regularization_reweighted
#seed=0
#for l in 0.05 # 0.01 0.005 0.001
#do
#    for num_epochs in 25
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} model.reference_regularization=${l} model.reinit=True"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done
#
#experiment=simclr_waterbirds_linear
#seed=0
#for l in 2 4
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} architecture.width_mult=${l}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#experiment=waterbirds_reference_regularization_annealing
#seed=0
#for l in 0.5 # 0.1 0.05
#do
#    for num_epochs in 5 # 10 25
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} model.reference_regularization=${l}"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done

#experiment=waterbirds_reweighted
#seed=0
#for num_epochs in 300
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} optimizer.lr=0.00001 optimizer.weight_decay=1"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#num_epochs=50
#experiment=waterbirds_logit_adjusted_loss
#seed=0
#
#for temperature in 1., 2., 3., 0.5; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs}
#    loss_fn.temperature=${temperature}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#num_epochs=50
#experiment=waterbirds_reweighted
#seed=0
#
#${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} optimizer.weight_decay=0"\
#-a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"


#experiment=waterbirds_reference_regularization_reweighted
#seed=0
#num_epochs=50
#for l in 1 10 20 40 # 10. 5. 2. #1.0 0.01 0.001  #0  1000 
#do
#    for lr in 0.0001
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} model.reference_regularization=${l} model.regularization_type=weights optimizer.lr=${lr}"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done
#experiment=waterbirds_reference_regularization
#seed=0
#for l in 20 50 100
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=25 model.regularization_type=logits model.reference_regularization=${l}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#for experiment in celeba_reweighted celeba_undersampled celeba_erm 
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#seed=0
#for experiment in celeba_undersampled 
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.checkpoint_callback=True"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#experiment=celeba_annealing
#seed=0
#for num_epochs in 50 100 150
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#experiment=waterbirds_reference_regularization_reweighted
#seed=0
#num_epochs=25
#for l in 10 # 0 1 10
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=${num_epochs} model.reference_regularization=${l} model.reinit=False model.only_update_misclassified=True optimizer.lr=0.0001"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#seed=0
#for experiment in civilcomments_resampled
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done
#
#for experiment in civilcomments_undersampled
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.checkpoint_callback=True"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#for experiment in civilcomments_dont_update_correct_extras
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#seed=0
#for experiment in waterbirds_dont_update_correct_extras
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.weight_decay=1.0 optimizer.lr=0.00001"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


# 2021-05-25
#num_epochs=20
#experiment=waterbirds_annealing
#seed=0
#for t in 0 2 4 6; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.num_epochs=10 datamodule.t=${t} trainer.max_epochs=${num_epochs}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#num_epochs=20
#experiment=waterbirds_annealing
#seed=0
#for t in 0 2 4 6 8; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.num_epochs=10 datamodule.t=${t} datamodule.reweigh=False trainer.max_epochs=${num_epochs}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#seed=0
#for experiment in waterbirds_dont_update_correct_extras #waterbirds_erm waterbirds_undersampled
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=True load_wandb_run=\"[kealexanderwang, importance-reweighing, waterbirds-undersampled-493, last.ckpt]\""\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#for experiment in waterbirds_dont_update_correct_extras
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=True load_wandb_run=\"[kealexanderwang, importance-reweighing, waterbirds-undersampled-493, last.ckpt]\" optimizer.momentum=0 trainer.max_epochs=100"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#experiment=waterbirds_dont_update_correct_extras
#for lr in 0.00001 0.000001 0.001 0.0001 
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=True load_wandb_run=\"[kealexanderwang, importance-reweighing, waterbirds-undersampled-493, last.ckpt]\" optimizer.momentum=0 optimizer.lr=${lr} trainer.max_epochs=100"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#experiment=waterbirds_dont_update_correct_extras
#for lr in 0.00001 0.000001 #0.001 0.0001 
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} load_wandb_run=\"[kealexanderwang, importance-reweighing, waterbirds-undersampled-493, last.ckpt]\" optimizer.momentum=0 optimizer.weight_decay=0 optimizer.lr=${lr} trainer.max_epochs=100"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#experiment=civilcomments_dont_update_correct_extras
#for lr in 0.00001 0.000001 #0.001 0.0001 
#do
#    for batch_size in 16 32
#    do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=True optimizer=sgd optimizer.momentum=0 optimizer.weight_decay=0 optimizer.lr=${lr} trainer.max_epochs=5 datamodule.batch_size=${batch_size}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done


#seed=0
#for lr in 0.0001 
#do
#    for experiment in waterbirds_erm 
#    do
#        for freeze_features in False #True
#        do
#            ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=${freeze_features} load_wandb_run=\"[kealexanderwang, importance-reweighing, waterbirds-undersampled-493, last.ckpt]\" optimizer.momentum=0 optimizer.lr=${lr} optimizer.weight_decay=0 trainer.max_epochs=100"\
#                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#        done
#    done
#done


#seed=0
#for lr in 0.0001 
#do
#    for experiment in waterbirds_erm 
#    do
#        for freeze_features in False #True
#        do
#            ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=${freeze_features} load_wandb_run=\"[kealexanderwang, importance-reweighing, waterbirds-undersampled-493, last.ckpt]\" optimizer.momentum=0 optimizer.lr=${lr} optimizer.weight_decay=0 trainer.max_epochs=100"\
#                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#        done
#    done
#done


#seed=0
#for reg in 5 10 20 30 50 100
#do
#    for experiment in waterbirds_reference_regularization
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.momentum=0 optimizer.lr=0.0001 optimizer.weight_decay=0 model.reference_regularization=${reg} trainer.max_epochs=50"\
#                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done


#seed=0
#for reg in 5
#do
#    for experiment in waterbirds_reference_regularization
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.momentum=0 optimizer.lr=0.0001 optimizer.weight_decay=0 model.reference_regularization=${reg} trainer.max_epochs=50"\
#                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done

#seed=0
#for reg in 10000
#do
#    for experiment in waterbirds_reference_regularization
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.momentum=0 optimizer.lr=0.0001 optimizer.weight_decay=0 model.reference_regularization=${reg} trainer.max_epochs=50"\
#                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done


#seed=0
#for reg in 0
#do
#    for experiment in waterbirds_reference_regularization
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.momentum=0 optimizer.lr=0.0001 optimizer.weight_decay=0 model.reference_regularization=${reg} trainer.max_epochs=50"\
#                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done


#seed=0
#for reg in 0.1 1 5 10 20
#do
#    for experiment in waterbirds_reference_regularization_reweighted
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.momentum=0 optimizer.lr=0.0001 optimizer.weight_decay=0 model.reference_regularization=${reg} trainer.max_epochs=50"\
#                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done


#seed=0
#for reg in 0.00001 0.0001 0.001 0.01 0.1 1 10
#do
#    for experiment in waterbirds_reference_regularization_reweighted
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.momentum=0 optimizer.lr=0.0001 optimizer.weight_decay=0 model.reference_regularization=${reg} load_wandb_run=null trainer.max_epochs=50"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done

# 2021-05-26
# TODO: these didn't launch
#seed=1
#for reg in 0.01 10
#do
#    for experiment in waterbirds_reference_regularization_reweighted
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} optimizer.momentum=0 optimizer.lr=0.0001 optimizer.weight_decay=0 model.reference_regularization=${reg}load_wandb_run=null trainer.max_epochs=50"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done

#seed=0
#for experiment in mnli_erm
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done              


#seed=0
#for experiment in mnli_undersampled   
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.checkpoint_callback=True"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

# 2021-05-27

#seed=0
#for experiment in mnli_undersampled   
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.train_frac=0.04 trainer.checkpoint_callback=True"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#seed=0
#for experiment in mnli_uneven_erm
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done
#
#seed=0
#for experiment in mnli_uneven_undersampled
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.checkpoint_callback=True"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#for experiment in \
#    waterbirds_reweighted \
#    waterbirds_erm \
#    waterbirds_resampled \
#    waterbirds_undersampled \
#;do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=True" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#for experiment in \
#    celeba_reweighted \
#    celeba_erm \
#    celeba_resampled \
#    celeba_undersampled \
#;do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=True" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#seed=0
#for experiment in mnli_uneven_dont_update_correct_extras
#do
#    for freeze in True False
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=${freeze}"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done


# TODO: rename the tag in wandb!
#seed=0
#experiment=mnli_uneven_erm
#for wrapper in resampled reweighted
#do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${wrapper}"\
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


#seed=0
#for experiment in \
#    celeba_resampled \
#;do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

# 2021-05-28

#seed=0
#for experiment in celeba_dont_update_correct_extras
#do
#    for freeze in True False
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} model.freeze_features=${freeze}"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    done
#done

# 2021-05-29
#seed=0
#experiment=cifar_exp_100
#for method in null undersampled resampled reweighted; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method}" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

#seed=0
#experiment=cifar_exp_100
#for method in null; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=none" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

# Test out kuangliu cifar settings without cosine annealing
#seed=0
#experiment=cifar_exp_100
#for method in null; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=none optimizer.lr=0.1 optimizer.weight_decay=0.0005" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


# Test out kuangliu cifar settings with cosine annealing
#seed=0
#experiment=cifar_exp_100
#for method in null; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=none optimizer.lr=0.1 optimizer.weight_decay=0.0005 +lr_scheduler=cosine_annealing" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done

# 2021-06-01
# Test out kuangliu cifar settings with cosine annealing AND data augmentation
#seed=0
#experiment=cifar_exp_100
#for method in null; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=none datamodule.data_augmentation=True optimizer.lr=0.1 optimizer.weight_decay=0.0005 +lr_scheduler=cosine_annealing" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


# Now make batch size match
#seed=0
#experiment=cifar_exp_100
#for method in null; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=none datamodule.data_augmentation=True datamodule.batch_size=128 optimizer.lr=0.1 optimizer.weight_decay=0.0005 +lr_scheduler=cosine_annealing" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


# Now make max_epochs match. This should match everything in kuangliu cifar repo
#seed=0
#experiment=cifar_exp_100
#for method in null; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} trainer.max_epochs=200 datamodule.wrapper_type=${method} datamodule.imb_type=none datamodule.data_augmentation=True datamodule.batch_size=128 optimizer.lr=0.1 optimizer.weight_decay=0.0005 +lr_scheduler=cosine_annealing" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#done


# 2021-06-02
# Try step imbalance on CIFAR10. Previously only had exp, 100
#seed=0
#experiment=cifar_exp_100
#for imb_factor in 10 100; do
#   for method in null undersampled resampled reweighted; do
#       ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=step datamodule.imb_factor=${imb_factor}" \
#           -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#       sleep 1
#   done
#done

# Try exp imbalance with 10:1 ratio
#seed=0
#experiment=cifar_exp_100
#for imb_factor in 10; do
#   for method in null undersampled resampled reweighted; do
#       ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=exp datamodule.imb_factor=${imb_factor}" \
#           -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#       sleep 1
#   done
#done


# 2021-06-03
# these experiments were missing val_worst_group_acc
#for experiment in \
#    waterbirds_reweighted \
#    waterbirds_erm \
#    waterbirds_resampled \
#
#do
#    for seed in 0
#    do
#        ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
#            -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#        sleep 1
#    done
#done

#seed=0
#experiment=cifar_exp_100
##notes="\\\'Turn off L2 regularization to see if reweighted and resampling revert to ERM. Out of all CIFAR10, only exp10 has reweighting work past 100% train acc\\\'"
#notes=""
#for method in resampled reweighted; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=${method} optimizer.weight_decay=0 datamodule.imb_factor=10" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    sleep 1
#done


# Try stronger weight decay to see if you can replace early stopping
#seed=0
#experiment=cifar_exp_100
#notes="stronger-l2"
#method=reweighted
#for weight_decay in 0.1 0.05 0.01 0.005 0.001; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=${method} optimizer.weight_decay=${weight_decay}" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    sleep 1
#done

# 2021-06-04
# some runs from before crashed
#seed=0
#experiment=cifar_exp_100
#notes="stronger-l2"
#method=reweighted
#for weight_decay in 0.05 0.01 0.005; do
#    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=${method} optimizer.weight_decay=${weight_decay}" \
#        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
#    sleep 1
#done

# 2021-07-07
# Testing out single quotes in commands
#seed=0
#experiment=waterbirds_undersampled
#notes="'Test test test'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes}\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}

# 2021-07-08
#seed=0
#experiment=celeba_erm
#notes="'Flood loss'"
#for flood_level in 0.01 0.05 0.1; do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} model.flood_level=${flood_level}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done

# 2021-07-22
#seed=0
#experiment=cifar_binary
#notes="'byrd-lipton reweighted convnet'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#seed=0
#experiment=cifar_binary
#notes="'byrd-lipton reweighted linear-model'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted architecture=linear_net datamodule.flatten_input=True\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#seed=0
#experiment=cifar_binary
#notes="'byrd-lipton reweighted convnet flood-loss'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} model.flood_level=0.05 datamodule.wrapper_type=reweighted\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#seed=0
#experiment=cifar_binary
#notes="'byrd-lipton reweighted linear-model flood-loss'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} model.flood_level=0.05 datamodule.wrapper_type=reweighted architecture=linear_net datamodule.flatten_input=True\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1


#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted convnet poly'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted linear-model poly'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted architecture=linear_net datamodule.flatten_input=True\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted convnet poly linear-type'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted loss_fn.type=linear\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted linear-model poly linear-type'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted architecture=linear_net datamodule.flatten_input=True loss_fn.type=linear\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

# exp-type loss will crash
#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted convnet poly exp-type'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted loss_fn.type=exp\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted linear-model poly exp-type'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted architecture=linear_net datamodule.flatten_input=True loss_fn.type=exp\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted convnet poly linear-type divide-loss-by-10'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted loss_fn.type=linear\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton erm linear-model poly linear-type'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} architecture=linear_net datamodule.flatten_input=True loss_fn.type=linear\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton erm convnet poly linear-type'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} loss_fn.type=linear\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#seed=0
#experiment=cifar_binary
#notes="'byrd-lipton erm convnet'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes}\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#seed=0
#experiment=cifar_binary
#notes="'byrd-lipton erm linear-model'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} architecture=linear_net datamodule.flatten_input=True\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted convnet poly linear-type masking'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted loss_fn.type=linear\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted convnet poly linear-type masking'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted loss_fn.type=linear\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1


#seed=0
#experiment=poly_cifar_binary
#notes="'byrd-lipton reweighted convnet poly linear-type lr-scaling-instead-of-loss'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.wrapper_type=reweighted loss_fn.type=linear optimizer.lr=0.01\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#CMD="${mynlprun} \"python run.py +experiment=poly_cifar_binary seed=0 trainer.max_epochs=2000 optimizer.lr=0.05 datamodule.batch_size=1024 datamodule.wrapper_type=reweighted datamodule.imb_factor=5 loss_fn.type=logit loss_fn.alpha=2\" \
#    -a is -g 1 -n \"$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

# 2021-08-11

# NOTE: SAVE
# Regular batch sizes
#seed=0
#experiment=waterbirds_reweighted
#notes="'tune lr'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_waterbirds_reweighted
#notes="'tune lr'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=celeba_reweighted
#notes="'tune lr'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_celeba_reweighted
#notes="'tune lr'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=cifar_binary_reweighted
#notes="'tune lr'"
#for lr in 0.1 0.05 0.01
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_cifar_binary_reweighted
#notes="'tune lr'"
#for lr in 0.1 0.05 0.01
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done

# Large batch
#seed=0
#experiment=waterbirds_reweighted
#notes="'tune lr'"
#acc=8
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_waterbirds_reweighted
#notes="'tune lr'"
#acc=8
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=celeba_reweighted
#notes="'tune lr'"
#acc=8
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_celeba_reweighted
#notes="'tune lr'"
#acc=8
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=cifar_binary_reweighted
#notes="'tune lr'"
#acc=8
#for lr in 0.1 0.05 0.01
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_cifar_binary_reweighted
#notes="'tune lr'"
#acc=8
#for lr in 0.1 0.05 0.01
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


# 2021-08-19
#seed=0
#experiment=cifar_binary_undersampled
#notes="'undersampled feature learning'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes}\" \
#    -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1

#seed=0
#notes="'frozen undersampled features'"
#wandb_run="'[kealexanderwang, importance-reweighing, stellar-morning-973, last.ckpt]'"
#lr=0.01
#for experiment in cifar_binary_reweighted poly_cifar_binary_reweighted
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} model.freeze_features=True load_wandb_run=${wandb_run}\" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done

# 2021-08-26
#seed=0
#notes="'try resampling every epoch'"
#for sampler in weighted null
#do
#    for experiment in cifar_binary_reweighted poly_cifar_binary_reweighted
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.train_sampler=${sampler}\" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done

#seed=0
#experiment=poly_cifar_binary_reweighted
#notes="'tune weight exponent for poly cifar'"
#for exponent in 1.0 1.2 1.4 1.6
#do
#   CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} model.weight_exponent=${exponent}\" \
#       -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#   eval ${CMD}
#   sleep 1
#done


#seed=0
#notes="'try dro'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in poly_waterbirds_dro poly_cifar_binary_dro
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes}\" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done

# 2021-08-30
#seed=0
#notes="'try dro with cross_entropy loss'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in cifar_binary_dro waterbirds_dro 
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} \" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


# 2021-09-02
#seed=0
#experiment=celeba_reweighted
#notes="'subsample celeba'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} \" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_celeba_reweighted
#notes="'subsample celeba'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} \" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#seed=0
#notes="'tune resampling every epoch'"
#for sampler in weighted null
#do
#    for experiment in cifar_binary_reweighted poly_cifar_binary_reweighted
#    do
#        for lr in 0.001 0.0005 0.0001
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} datamodule.train_sampler=${sampler}\" \
#                -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#seed=0
#notes="'dro imagenet init'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in waterbirds_dro 
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} architecture.pretrained=True\" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done
#
#
#seed=0
#notes="'dro imagenet init'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in poly_waterbirds_dro
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} architecture.pretrained=True\" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done

#seed=0
#experiment=celeba_reweighted
#notes="'subsample celeba'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} \" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_celeba_reweighted
#notes="'subsample celeba'"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} \" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done

# TODO: revert to old loss before running DRO

# 2021-09-08
# 2021-09-09
#group="'2021-09-09_adversarial-dro'"
#
#seed=0
#notes="'dro random init'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in poly_waterbirds_dro poly_cifar_binary_dro
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} optimizer.lr=${lr}\" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done
#
#seed=0
#notes="'dro random init'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in cifar_binary_dro waterbirds_dro 
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} optimizer.lr=${lr}\" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done
#
#
#
#seed=0
#notes="'dro imagenet init'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in waterbirds_dro 
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} optimizer.lr=${lr} architecture.pretrained=True \" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done
#
#
#seed=0
#notes="'dro imagenet init'"
#for lr in 0.0001 0.0005 0.001
#do
#    for experiment in poly_waterbirds_dro
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} optimizer.lr=${lr} architecture.pretrained=True\" \
#            -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


#group="'2021-09-09_celeba-subsampled'"
#seed=0
#experiment=celeba_reweighted
#notes="''"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} optimizer.lr=${lr} \" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#seed=0
#experiment=poly_celeba_reweighted
#notes="''"
#for lr in 0.001 0.0005 0.0001
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} optimizer.lr=${lr} \" \
#        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done



#group="'2021-09-20_celeba-subsampled'"
#notes="'2021-09-09 celeba gave bad test acc for poly loss. Rerunning with imagenet init to see if gap narrows'"
#experiment=celeba_reweighted
#for seed in 0
#do
#    for loss_fn in cross_entropy polynomial_loss
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} architecture.pretrained=True\" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done

# 2021-09-20
#group="'2021-09-20_celeba'"
#notes="'Train with 0.0004 lr for now. Will sweep hypers again later'"
#for seed in 0 1 2 3 4
#do
#    for experiment in celeba_erm celeba_reweighted celeba_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done
#
#
#group="'2021-09-20_cifar'"
#notes="''"
#for seed in 0 1 2 3 4
#do
#    for experiment in cifar_erm cifar_reweighted cifar_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


# 2021-09-22
#group="'2021-09-22_cifar_debug'"
#notes="'Sweep lr for cifar to check why old poly loss is not interpolating'"
##
##seed=0
##for lr in 0.01 0.005 0.001 0.0005 0.0001
##do
##    for experiment in cifar_reweighted
##    do
##        for loss_fn in polynomial_loss
##        do
##            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr}\" \
##                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
##            eval ${CMD}
##            sleep 1
##        done
##    done
##done
#
#seed=0
#momentum=0.
#for lr in 0.01
#do
#    for experiment in cifar_reweighted
#    do
#        for loss_fn in polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr} optimizer.momentum=${momentum}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-09-22_cifar'"
#notes="'Lowered learning rate of poly loss in order to interpolate. See Group 2021-09-20_cifar'"
#for seed in 0 1 2 3 4
#do
#    for experiment in cifar_erm cifar_reweighted cifar_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy #polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-09-22_celeba_dro_sweep'"
#notes="'Sweep LR for celeba dro'"

#seed=0
#for lr in 0.0004 0.00004 0.00001
#do
#    for experiment in celeba_dro
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done

#group="'2021-09-22_cifar_dro_sweep'"
#notes="'Sweep LR for cifar dro'"
#seed=0
#for lr in 0.001 0.0005 0.0001
#do
#    for experiment in cifar_dro
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#2021-09-26
#group="'2021-09-26_celeba_dro'"
#notes="'train celeba dro for longer'"
#num_epochs=100
#lr=0.00004
#for seed in 0 1 2 3 4
#do
#    for experiment in celeba_dro
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


# 2021-09-29
#group="'2021-09-29_celeba_dro'"
#notes="'train celeba dro for even longer'"
#num_epochs=200
#lr=0.00004
#for seed in 0 1 2 3 4
#do
#    for experiment in celeba_dro
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


# 2021-10-01
#group="'2021-10-01_celeba'"
#notes="'Fix train val test seed for celeba'"
#for seed in 0 1 2 3 4
#do
#    for experiment in celeba_erm celeba_reweighted celeba_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-10-01_celeba_2'"
#notes="'Fix train val test seed for celeba and seed 0 for train set'"
##for seed in 0 1 2 3 4
#for seed in 5 6 7 8 9
#do
#    for experiment in celeba_erm celeba_reweighted celeba_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


# 2021-10-03
#group="'2021-10-03_celeba_rw2x'"
#notes="'Officially integrate weight multiplier'"
#multiplier=2
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    for experiment in celeba_reweighted celeba_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} model.weight_multiplier=${multiplier}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done
## Note this is different from the actual doubling scheme done by Nildari!


#group="'2021-10-01_celeba_2'"
#notes="'Rerun because of NaN for RW ES'"
#for seed in 8
#do
#    for experiment in celeba_reweighted_early_stopped
#    do
#        for loss_fn in polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done

#group="'2021-10-03_cifar_scaled_up'"
#notes="'Redo niladris scaled up cifar runs to have in same project'"
#lr=0.01
#momentum=0
#for seed in 0 1 2 3 4
#do
#    for experiment in cifar_reweighted cifar_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            for multiplier in 2 32
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} model.weight_multiplier=${multiplier} optimizer.lr=${lr} optimizer.momentum=${momentum}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done
## All runs above failed because nildari used different doubling scheme



#group="'2021-10-03_exponent'"
#notes="'Try weight exponentiation'"
#for seed in 0 1 2
#do
#    for experiment in celeba_reweighted cifar_reweighted
#    do
#        for loss_fn in polynomial_loss cross_entropy 
#        do
#            for exponent in 1.0 2.0 3.0
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done


#group="'2021-10-03_celeba_exponent'"
#notes="'weight exponentiation'"
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    for experiment in celeba_reweighted celeba_reweighted_early_stopped
#    do
#        for loss_fn in polynomial_loss cross_entropy 
#        do
#            for exponent in 2.0
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done


#group="'2021-10-03_sweep_cifar_exponent'"
#notes="'weight exponentiation'"
#loss_fn=polynomial_loss
#experiment=cifar_reweighted
#for seed in 0 1
#do
#    for momentum in 0. 0.9
#    do
#        for lr in 0.01 0.001 0.0001
#        do
#            for exponent in 1.5 2.0
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done


#group="'2021-10-03_cifar_exponent'"
#notes="'weight exponentiation'"
#num_epochs=400
#momentum=0.
#lr=0.008
#for seed in 0 1 2 3 4
#do
#    for experiment in cifar_reweighted cifar_reweighted_early_stopped
#    do
#        for loss_fn in polynomial_loss cross_entropy 
#        do
#            for exponent in 1.5 2.0
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done


#group="'2021-10-03_celeba_exponent'"
#notes="'Run cross entropy for longer and see if it is worse'"
#num_epochs=100
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    for experiment in celeba_reweighted
#    do
#        for loss_fn in cross_entropy 
#        do
#            for exponent in 2.0
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done


#group="'2021-10-03_celeba_exponent'"
#notes="'Run for longer'"
#num_epochs=100
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    for experiment in celeba_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy 
#        do
#            for exponent in 2.0
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done
#
#
#group="'2021-10-03_celeba_exponent'"
#notes="'Run for longer'"
#num_epochs=100
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    for experiment in celeba_reweighted celeba_reweighted_early_stopped
#    do
#        for loss_fn in polynomial_loss
#        do
#            for exponent in 2.0
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done


# 2021-10-04
#group="'2021-09-22_cifar'"
#notes="'Rerun failed run'"
#for seed in 4
#do
#    for experiment in cifar_reweighted
#    do
#        for loss_fn in polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-10-04_cifar'"
#notes="'Train cifar for longer to interpolate'"
#num_epochs=400
#for seed in 0 1 2 3 4
#do
#    for experiment in cifar_erm cifar_reweighted cifar_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-09-22_cifar'"
#notes="'Train ERM cross entropy for longer to interpolate'"
#num_epochs=500
#for seed in 0 1 2 3 4
#do
#    for experiment in cifar_erm 
#    do
#        for loss_fn in cross_entropy
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-10-04_cifar'"
#notes="'Rerun crashed runs'"
#num_epochs=400
#for seed in 0 1
#do
#    for experiment in cifar_reweighted 
#    do
#        for loss_fn in polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done
#
#
#group="'2021-10-04_cifar'"
#notes="'Rerun crashed runs'"
#num_epochs=400
#for seed in 1
#do
#    for experiment in cifar_reweighted_early_stopped
#    do
#        for loss_fn in polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done
#
#
#group="'2021-10-04_cifar'"
#notes="'Rerun crashed runs'"
#num_epochs=400
#for seed in 0 1
#do
#    for experiment in cifar_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-10-04'"
#notes="'Train xent with higher lr and epochs'"
#num_epochs=200
#lr=0.01
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    for experiment in cifar_erm 
#    do
#        for loss_fn in cross_entropy
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-10-04_cifar'"
#notes="'5 new seeds'"
#num_epochs=400
#for experiment in cifar_erm cifar_reweighted cifar_reweighted_early_stopped
#do
#    for seed in 5 6 7 8 9
#    do
#        for loss_fn in cross_entropy
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done

#group="'2021-10-04_cifar'"
#notes="'5 new seeds'"
#num_epochs=400
#for experiment in cifar_erm cifar_reweighted cifar_reweighted_early_stopped
#do
#    for seed in 5 6 7 8 9
#    do
#        for loss_fn in polynomial_loss
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done
#
#
#group="'2021-10-03_cifar_exponent'"
#notes="'weight exponentiation'"
#num_epochs=400
#momentum=0.
#lr=0.008
#for seed in 5 6 7 8 9
#do
#    for experiment in cifar_reweighted cifar_reweighted_early_stopped
#    do
#        for loss_fn in polynomial_loss cross_entropy 
#        do
#            for exponent in 1.5
#            do
#                CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#                eval ${CMD}
#                sleep 1
#            done
#        done
#    done
#done


#group="'2021-10-04_cifar'"
#notes="'10 new seeds'"
#num_epochs=400
#for experiment in cifar_erm
#do
#    for seed in 10 11 12 13 14 15 16 17 18 19
#    do
#        for loss_fn in cross_entropy
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-10-04_cifar'"
#notes="'Higher LR for all xent runs'"
#num_epochs=300
#lr=0.01
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    for experiment in cifar_erm cifar_reweighted cifar_reweighted_early_stopped
#    do
#        for loss_fn in cross_entropy
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done

#group="'2021-11-09_vsloss'"
#notes="'tune vsloss'"
#num_epochs=300
#lr=0.01
#seed=0
#loss_fn=vs_loss
#num_per_class="'[4000, 400]'"
#for tau in 0.5 0.75 1.0 1.25 1.5
#do
#    for gamma in 0.05 0.1 0.15 0.2
#    do
#        for experiment in cifar_erm
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#                -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-11-10_vsloss'"
#notes="'train vsloss on best settings'"
#num_epochs=300
#lr=0.01
#loss_fn=vs_loss
#experiment=cifar_erm
#num_per_class="'[4000, 400]'"
#tau=1.0
#gamma=0.05
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-11-09_laloss'"
#notes="'tune laloss'"
#num_epochs=300
#lr=0.01
#seed=0
#experiment=cifar_erm
#loss_fn=vs_loss
#num_per_class="'[4000, 400]'"
#gamma=0.
#for tau in 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75. 3.0
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#
#group="'2021-11-09_cdtloss'"
#notes="'tune cdtloss'"
#num_epochs=300
#lr=0.01
#seed=0
#experiment=cifar_erm
#loss_fn=vs_loss
#num_per_class="'[4000, 400]'"
#tau=0.
#for gamma in 0.1 0.2 0.3 0.4 0.5
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-11-10_laloss'"
#notes="'train laloss on best settings'"
#num_epochs=300
#lr=0.01
#loss_fn=vs_loss
#experiment=cifar_erm
#num_per_class="'[4000, 400]'"
#tau=2.5
#gamma=0.
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#
#group="'2021-11-10_cdtloss'"
#notes="'train cdtloss on best settings'"
#num_epochs=300
#lr=0.01
#loss_fn=vs_loss
#experiment=cifar_erm
#num_per_class="'[4000, 400]'"
#tau=0.
#gamma=0.4
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-11-10_laloss_lb_longer'"
#notes="'train laloss on best settings with larger batch size but train 3x longer since 300 epochs is not enough to interpolate fully'"
#num_epochs=900
#lr=0.01
#loss_fn=vs_loss
#experiment=cifar_erm
#num_per_class="'[4000, 400]'"
#batch_size=512
#tau=2.5
#gamma=0.
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr} datamodule.batch_size=${batch_size}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-10-01_vsloss_celeba'"
#notes="'Tune VSGroupLoss on CelebA'"
#experiment=celeba_erm 
#seed=0
#loss_fn=vs_group_loss
#num_per_group="'[1446,1308,468,33]'"
#for gamma in 0.1 0.2 0.3 0.4 0.5
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.gamma=${gamma} loss_fn.num_per_group=${num_per_group}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-10-01_vsloss_celeba'"
#notes="'Tune VSGroupLoss on CelebA'"
#experiment=celeba_erm 
#seed=0
#loss_fn=vs_group_loss
#num_per_group="'[1446,1308,468,33]'"
#for gamma in 0.6 0.7 0.8 0.9 1.0
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.gamma=${gamma} loss_fn.num_per_group=${num_per_group}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-11-11_vsloss_celeba'"
#notes="'train VSGroupLoss on CelebA with best hypers'"
#experiment=celeba_erm 
#loss_fn=vs_group_loss
#num_per_group="'[1446,1308,468,33]'"
#gamma=0.4
##for seed in 0 1 2 3 4 5 6 7 8 9
#for seed in 8
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.gamma=${gamma} loss_fn.num_per_group=${num_per_group}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-11-11_tune_vsloss_again'"
#notes="'tune vsloss again except we cover laloss and cdt in our grid search'"
#num_epochs=300
#lr=0.01
#seed=0
#loss_fn=vs_loss
#num_per_class="'[4000, 400]'"
#experiment=cifar_erm
#for tau in 0.0 0.5 1.0 1.5 2.0 2.5 3.0
#do
#    for gamma in 0.0 0.1 0.2 0.3 0.4 0.5
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


#group="'2021-11-11_tune_ldam'"
#notes="'tune ldam'"
#num_epochs=300
#lr=0.01
#seed=0
#loss_fn=ldam
#num_per_class="'[4000, 400]'"
#experiment=cifar_erm
#for max_margin in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.max_margin=${max_margin} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-11-11_best_ldam'"
#notes="'best ldam'"
#num_epochs=300
#lr=0.01
#max_margin=1.0
#loss_fn=ldam
#num_per_class="'[4000, 400]'"
#experiment=cifar_erm
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.max_margin=${max_margin} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done

#group="'2021-11-11_best_vsloss'"
#notes="'best vsloss'"
#num_epochs=300
#lr=0.01
#tau=3.0
#gamma=0.3
#loss_fn=vs_loss
#num_per_class="'[4000,400]'"
#experiment=cifar_erm
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#group="'2021-11-11_best_laloss'"
#notes="'best laloss'"
#num_epochs=300
#lr=0.01
#tau=3.0
#gamma=0.0
#loss_fn=vs_loss
#num_per_class="'[4000,400]'"
#experiment=cifar_erm
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#group="'2021-11-11_best_cdtloss'"
#notes="'best cdtloss'"
#num_epochs=300
#lr=0.01
#tau=0.0
#gamma=0.5
#loss_fn=vs_loss
#num_per_class="'[4000, 400]'"
#experiment=cifar_erm
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done



#group="'2021-11-11_best_laloss_longer'"
#notes="'best laloss trained for longer'"
#num_epochs=900
#lr=0.01
#tau=3.0
#gamma=0.0
#loss_fn=vs_loss
#num_per_class="'[4000,400]'"
#experiment=cifar_erm
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#
#group="'2021-11-11_best_vsloss_longer'"
#notes="'best vsloss trained for longer'"
#num_epochs=900
#lr=0.01
#tau=3.0
#gamma=0.3
#loss_fn=vs_loss
#num_per_class="'[4000,400]'"
#experiment=cifar_erm
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} trainer.max_epochs=${num_epochs} optimizer.lr=${lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


#group="'2021-11-12_tune_cifar_exponent_and_alpha'"
#notes="'tune alpha and weight exponent     old runs didnt tune alpha'"
#num_epochs=900
#momentum=0.
#lr=0.008
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#for seed in 0
#do
#    for alpha in 0.25 0.5 1.0 2.0 4.0
#    do
#        for exponent in 1.5 2.0 2.5 3.0
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done
#
#num_epochs=300
#momentum=0.9
#lr=0.01
#for seed in 0 1 2
#do
#    for alpha in 0.25 0.5 1.0 2.0 4.0
#    do
#        for exponent in 1.0
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-11-12_celeba_dro'"
#notes="'train celeba after fixing dro higher lr to interpolate'"
#num_epochs=200
#lr=0.001
#experiment=celeba_dro
#seed=0
#for adv_probs_lr in 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.1
#do
#    for loss_fn in cross_entropy polynomial_loss
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} model.adv_probs_lr=${adv_probs_lr}\" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done

#group="'2021-11-12_tune_cifar_alpha_squared'"
#notes="'tune exponent for alpha equal to 2'"
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#num_epochs=600
#momentum=0.
#lr=0.005
#alpha=2.0
#for exponent in 2.5 3.0
#do
#    for seed in 0 1 2 3 4 5 6 7 8 9
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


#group="'2021-11-12_celeba_dro'"
#notes="'train celeba after fixing dro higher lr to interpolate'"
#num_epochs=200
#lr=0.001
#experiment=celeba_dro
#seed=0
#for adv_probs_lr in 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.1
#do
#    for loss_fn in cross_entropy polynomial_loss
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} model.adv_probs_lr=${adv_probs_lr}\" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


#group="'2021-11-13_tune_cifar_alpha_squared'"
#notes="'tune exponent for alpha equal to 2 but train longer to interpolate'"
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#num_epochs=600
#momentum=0.
#lr=0.08
#alpha=2.0
#for exponent in 2.5 3.0
#do
#    for seed in 0 1 2 3 4 5 6 7 8 9
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


#group="'2021-11-13_celeba_dro_cross_entropy'"
#notes="'train celeba with dro for cross entropy'"
#num_epochs=200
#lr=0.001
#experiment=celeba_dro
#adv_probs_lr=0.05
#for loss_fn in cross_entropy
#do
#    for seed in 0 1 2 3 4 5 6 7 8 9
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} model.adv_probs_lr=${adv_probs_lr}\" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


group="'2021-11-13_celeba_dro_compare_poly_loss'"
notes="'train celeba with dro for poly loss with two settings'"
num_epochs=200
lr=0.001
experiment=celeba_dro
adv_probs_lr=0.05
loss_fn=polynomial_loss

#for alpha in 1.0 2.0
for alpha in 2.0
do
    #for seed in 0 1 2 3 4 5 6 7 8 9
    for seed in 2 3 7 9
    do
        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} model.adv_probs_lr=${adv_probs_lr}\" \
            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
        eval ${CMD}
        sleep 1
    done
done
#
#
#group="'2021-11-13_celeba_dro_vs_group_loss'"
#notes="'train celeba with dro for vs group loss'"
#num_epochs=200
#lr=0.001
#experiment=celeba_dro
#loss_fn=vs_group_loss
#adv_probs_lr=0.05
#num_per_group="'[1446,1308,468,33]'"
#gamma=0.4
#
#for seed in 0 1 2 3 4 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.gamma=${gamma} loss_fn.num_per_group=${num_per_group} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} model.adv_probs_lr=${adv_probs_lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


# Rerun due to nans
#group="'2021-11-13_tune_cifar_alpha_squared'"
#notes="'tune exponent for alpha equal to 2 but train longer to interpolate'"
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#num_epochs=600
#momentum=0.
#lr=0.08
#alpha=2.0
#for exponent in 2.5
#do
#    for seed in 3
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done


#group="'2021-11-13_tune_celeba_exponent_and_alpha'"
#notes="'tune alpha and weight exponent on celeba'"
#experiment=celeba_reweighted
#loss_fn=polynomial_loss
#for seed in 0
#do
#    for alpha in 1.0 2.0
#    do
#        for exponent in 1.0 1.5 2.0 2.5 3.0
#        do
#            CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} \" \
#            -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#            eval ${CMD}
#            sleep 1
#        done
#    done
#done


#group="'2021-11-13_celeba_best_polyloss'"
#notes="'try best hypers for poly loss on celeba'"
#experiment=celeba_reweighted
#loss_fn=polynomial_loss
#alpha=2.0
#exponent=2.5
#max_epochs=100
##for seed in 0 1 2 3 4 5 6 7 8 9
#for seed in 5 6 7 8 9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} \" \
#    -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#
#
#group="'2021-11-13_celeba_tune_dro_vs_group_loss'"
#notes="'tune dro vs group loss since last set of runs crashed'"
#num_epochs=200
#lr=0.001
#experiment=celeba_dro
#loss_fn=vs_group_loss
#adv_probs_lr=0.05
#num_per_group="'[1446,1308,468,33]'"
#seed=0
#for gamma in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.gamma=${gamma} loss_fn.num_per_group=${num_per_group} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} model.adv_probs_lr=${adv_probs_lr}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done


# Rerun due to nans
#group="'2021-11-13_tune_cifar_alpha_squared'"
#notes="'tune exponent for alpha equal to 2 but train longer to interpolate'"
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#num_epochs=600
#momentum=0.
#lr=0.08
#alpha=2.0
#for exponent in 3.0
#do
#    for seed in 3
#    do
#        CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#        eval ${CMD}
#        sleep 1
#    done
#done

#group="'2021-11-13_celeba_best_polyloss'"
#notes="'try best hypers for poly loss on celeba but train for longer past interpolation'"
#experiment=celeba_reweighted
#loss_fn=polynomial_loss
#alpha=2.0
#exponent=2.5
#num_epochs=200
##for seed in 0 1 2 3 4 5 6 7 8 9
#for seed in 3
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} trainer.max_epochs=${num_epochs}\" \
#        -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done

#========================================================================================================================
#notes="'use best hypers and train with increasing batch sizes while keeping number of gradient updates fixed'"
#group="'2021-11-15_increasing_batch_size_cifar10'"
#seed=0
#
#num_epochs=300
#batch_size=64
#
##experiment=cifar_reweighted
##loss_fn=polynomial_loss
##momentum=0.
##lr=0.08
##alpha=2.0
##exponent=2.5
##CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
##-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
##eval ${CMD}
##sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.01
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#batch_size=128
#
##experiment=cifar_reweighted
##loss_fn=polynomial_loss
##momentum=0.
##lr=0.11
##alpha=2.0
##exponent=2.5
##CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
##-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
##eval ${CMD}
##sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.0141
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#batch_size=256
#
##experiment=cifar_reweighted
##loss_fn=polynomial_loss
##momentum=0.
##lr=0.16
##alpha=2.0
##exponent=2.5
##CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
##-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
##eval ${CMD}
##sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.02
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#batch_size=512
#
##experiment=cifar_reweighted
##loss_fn=polynomial_loss
##momentum=0.
##lr=0.226
##alpha=2.0
##exponent=2.5
##CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
##-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
##eval ${CMD}
##sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.0283
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#========================================================================================================================


##========================================================================================================================
#notes="'use best hypers and train with increasing batch sizes while keeping number of gradient updates fixed and linear
#scaling lr with bs'"
#group="'2021-11-15_increasing_batch_size_cifar10_linear_scaling'"
#seed=0
#
#num_epochs=300
#batch_size=64
#
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#momentum=0.
#lr=0.08
#alpha=2.0
#exponent=2.5
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.01
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#batch_size=128
#
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#momentum=0.
#lr=0.16
#alpha=2.0
#exponent=2.5
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.02
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#batch_size=256
#
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#momentum=0.
#lr=0.32
#alpha=2.0
#exponent=2.5
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.04
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#
#
#batch_size=512
#
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#momentum=0.
#lr=0.64
#alpha=2.0
#exponent=2.5
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#experiment=cifar_erm
#loss_fn=vs_loss
#momentum=0.9
#lr=0.08
#tau=3.
#gamma=0.3
#num_per_class="'[4000,400]'"
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.tau=${tau} loss_fn.gamma=${gamma} loss_fn.num_per_class=${num_per_class} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
##========================================================================================================================


#========================================================================================================================
#notes="'use best hypers and train with increasing batch sizes while keeping number of gradient updates fixed'"
#group="'2021-11-15_increasing_batch_size_cifar10'"
#seed=0
#
#num_epochs=300
#batch_size=256
#
#experiment=cifar_reweighted
#loss_fn=polynomial_loss
#momentum=0.
#lr=0.32
#alpha=2.0
#exponent=2.5
#CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} loss_fn.alpha=${alpha} datamodule.train_weight_exponent=${exponent} optimizer.momentum=${momentum} optimizer.lr=${lr} trainer.max_epochs=${num_epochs} datamodule.batch_size=${batch_size}\" \
#-a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#eval ${CMD}
#sleep 1
#========================================================================================================================


#========================================================================================================================
#notes="'cifar undersampled for rebuttal'"
#group="'2021-11-15_cifar_undersampled'"
#
#experiment=cifar_undersampled
#loss_fn=cross_entropy
#epochs=600
#for seed in 0 1 2 3 4 5 6 7 8 9 
#do
#    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} logger.wandb.group=${group} loss_fn=${loss_fn} trainer.max_epochs=${epochs}\" \
#    -a is -g 1 -n \"${experiment}-${group}-$(date '+%d-%m-%Y_%H:%M:%S')\""
#    eval ${CMD}
#    sleep 1
#done
#========================================================================================================================
