#!/bin/bash
#mynlprun="nlprun -x jagupard[4-8],jagupard[19-20],jagupard[26-29]"
mynlprun="nlprun3 -x jagupard[19-20],jagupard[25-29]"

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

# NOTE: FINAL
# Regular batch sizes
seed=0
experiment=waterbirds_reweighted
notes="'tune lr'"
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=poly_waterbirds_reweighted
notes="'tune lr'"
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=celeba_reweighted
notes="'tune lr'"
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=poly_celeba_reweighted
notes="'tune lr'"
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=cifar_binary_reweighted
notes="'tune lr'"
for lr in 0.1 0.05 0.01
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=poly_cifar_binary_reweighted
notes="'tune lr'"
for lr in 0.1 0.05 0.01
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

# Large batch
seed=0
experiment=waterbirds_reweighted
notes="'tune lr'"
acc=8
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=poly_waterbirds_reweighted
notes="'tune lr'"
acc=8
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=celeba_reweighted
notes="'tune lr'"
acc=8
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=poly_celeba_reweighted
notes="'tune lr'"
acc=8
for lr in 0.001 0.0005 0.0001
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=cifar_binary_reweighted
notes="'tune lr'"
acc=8
for lr in 0.1 0.05 0.01
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done

seed=0
experiment=poly_cifar_binary_reweighted
notes="'tune lr'"
acc=8
for lr in 0.1 0.05 0.01
do
    CMD="${mynlprun} \"python run.py +experiment=${experiment} seed=${seed} logger.wandb.notes=${notes} optimizer.lr=${lr} trainer.accumulate_grad_batches=${acc}\" \
        -a is -g 1 -n \"${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')\""
    eval ${CMD}
    sleep 1
done
