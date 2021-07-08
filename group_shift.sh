#!/bin/bash
mynlprun="nlprun3 -x jagupard[19-20],jagupard[25-29]"
seed=0

# Group shift with spurious correlations
# Random init
for experiment in \
    waterbirds_reweighted \
    waterbirds_erm \
    waterbirds_undersampled \
    celeba_reweighted \
    celeba_erm \
    celeba_undersampled \

do
    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed} architecture.pretrained=False"\
        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
    sleep 1
done

# Group shift without spurious correlations
for experiment in \
    civilcomments_reweighted
    civilcomments_erm \
    civilcomments_undersampled \
    mnli_uneven_reweighted
    mnli_uneven_erm \
    mnli_uneven_undersampled \

do
    ${mynlprun} "python run.py +experiment=${experiment} seed=${seed}"\
        -a is -g 1 -n "${experiment}-${seed}-$(date '+%d-%m-%Y_%H:%M:%S')"
    sleep 1
done
