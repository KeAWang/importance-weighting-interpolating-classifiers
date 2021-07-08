#!/bin/bash
mynlprun="nlprun3 -x jagupard[19-20],jagupard[25-29]"

# Label shift
seed=0
experiment=cifar_exp_100
for imb_factor in 10 100; do
    for imb_type in exp step; do
        for method in null undersampled reweighted; do
            ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} datamodule.wrapper_type=${method} datamodule.imb_type=step datamodule.imb_factor=${imb_factor}" \
                -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
            sleep 1
        done
    done
done
