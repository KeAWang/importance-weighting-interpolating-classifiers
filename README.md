# Is Importance Weighting Incompatible with Interpolating Classifiers?  (ICLR 2022)

- [Code](https://github.com/KeAWang/importance-weighting-interpolating-classifiers)
- [ICLR 2022 OpenReview](https://openreview.net/forum?id=uqBOne3LUKy)
- [arXiv](https://arxiv.org/abs/2112.12986)
- Also appeared at [NeurIPS 2021 DistShift Workshop](https://sites.google.com/view/distshift2021) as a spotlight paper

This repo is created from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
Experiment settings are managed by [hydra](https://hydra.cc/), a hierarchical config framework, and the setting files are specified in the `configs/` directory.

## Setup instructions

0. Make a weights and bias account
1. `conda env create -f conda_env.yml`
2. `pip install requirements.txt`
3. Copy `.env.tmp` to a new file called `.env`. Edit the `PERSONAL_DIR` environment variable in `.env` to be the root directory of where you want to store your data.
4. Simplest command: `python run.py +experiment=celeba_erm`.

## Reproducing experiments

The commandline scripts below will run with the default seed. In our paper we loop over seeds for each experiment, which you can do by appending `seed=0,1,2,3,4,5,6,7,8,9` to the launch script below.

Some seeds may result in NaNs during training. Relaunching the experiments (without changing the seed) will get rid of the NaNs, likely due to GPU non-determinism.

### Figure 1

Run `notebooks/two-gaussians.ipynb`

### Figure 2

Run `notebooks/figure2_left.ipynb` and `notebooks/figure2_right.ipynb`

### Figure 3 (Interpolation experiments)

```bash
# Imbalanced Binary CIFAR10
python run.py +experiment=cifar_erm loss_fn=cross_entropy trainer.max_epochs=400
python run.py +experiment=cifar_reweighted loss_fn=cross_entropy trainer.max_epochs=400
python run.py +experiment=cifar_reweighted loss_fn=cross_entropy trainer.max_epochs=400 datamodule.train_weight_exponent=1.5 optimizer.momentum=0. optimizer.lr=0.008 

python run.py +experiment=cifar_erm loss_fn=polynomial_loss trainer.max_epochs=400
python run.py +experiment=cifar_reweighted loss_fn=polynomial_loss trainer.max_epochs=400
python run.py +experiment=cifar_reweighted loss_fn=polynomial_loss trainer.max_epochs=400 datamodule.train_weight_exponent=1.5 optimizer.momentum=0. optimizer.lr=0.008 

# Subsampled CelebA
python run.py +experiment=celeba_erm loss_fn=cross_entropy 
python run.py +experiment=celeba_reweighted loss_fn=cross_entropy 
python run.py +experiment=celeba_reweighted loss_fn=cross_entropy datamodule.train_weight_exponent=2.0 trainer.max_epochs=100

python run.py +experiment=celeba_erm loss_fn=polynomial_loss
python run.py +experiment=celeba_reweighted loss_fn=polynomial_loss
python run.py +experiment=celeba_reweighted loss_fn=polynomial_loss datamodule.train_weight_exponent=2.0 trainer.max_epochs=100
```

### Figure 4 (Early-stopped experiments)

```bash
# Imbalanced Binary CIFAR10

python run.py +experiment=cifar_reweighted loss_fn=cross_entropy trainer.max_epochs=400
python run.py +experiment=cifar_reweighted_early_stopped loss_fn=cross_entropy trainer.max_epochs=400
python run.py +experiment=cifar_reweighted_early_stopped loss_fn=cross_entropy trainer.max_epochs=400 datamodule.train_weight_exponent=1.5 optimizer.momentum=0. optimizer.lr=0.008 


python run.py +experiment=cifar_reweighted loss_fn=polynomial_loss trainer.max_epochs=400
python run.py +experiment=cifar_reweighted_early_stopped loss_fn=polynomial_loss trainer.max_epochs=400
python run.py +experiment=cifar_reweighted_early_stopped loss_fn=polynomial_loss trainer.max_epochs=400 datamodule.train_weight_exponent=1.5 optimizer.momentum=0. optimizer.lr=0.008 

# Subsampled CelebA
python run.py +experiment=celeba_reweighted loss_fn=cross_entropy 
python run.py +experiment=celeba_reweighted_early_stopped loss_fn=cross_entropy
python run.py +experiment=celeba_reweighted_early_stopped loss_fn=cross_entropy datamodule.train_weight_exponent=2.0 trainer.max_epochs=100

python run.py +experiment=celeba_reweighted loss_fn=polynomial_loss
python run.py +experiment=celeba_reweighted_early_stopped loss_fn=polynomial_loss
python run.py +experiment=celeba_reweighted_early_stopped loss_fn=polynomial_loss datamodule.train_weight_exponent=2.0 trainer.max_epochs=100

```

### Figure 5

```bash
# Imbalanced Binary CIFAR10
# Poly+IW
python run.py +experiment=cifar_reweighted loss_fn=polynomial_loss loss_fn.alpha=2.0 datamodule.train_weight_exponent=3.0 optimizer.momentum=0. optimizer.lr=0.08 trainer.max_epochs=600
# CE+US
python run.py +experiment=cifar_undersampled loss_fn=cross_entropy trainer.max_epochs=600
# LDAM
python run.py +experiment=cifar_erm loss_fn=ldam loss_fn.max_margin=1.0 loss_fn.num_per_class="[4000, 400]" trainer.max_epochs=300 optimizer.lr=0.01
# CDT
python run.py +experiment=cifar_erm loss_fn=vs_loss loss_fn.tau=0.0 loss_fn.gamma=0.5 loss_fn.num_per_class="[4000, 400]" trainer.max_epochs=300 optimizer.lr=0.01
# LA
python run.py +experiment=cifar_erm loss_fn=vs_loss loss_fn.tau=3.0 loss_fn.gamma=0.0 loss_fn.num_per_class="[4000,400]" trainer.max_epochs=300 optimizer.lr=0.01
# VS
python run.py +experiment=cifar_erm loss_fn=vs_loss loss_fn.tau=3.0 loss_fn.gamma=0.3 loss_fn.num_per_class="[4000,400]" trainer.max_epochs=300 optimizer.lr=0.01

# Subsampled CelebA
# Poly+IW
python run.py +experiment=celeba_reweighted loss_fn=polynomial_loss loss_fn.alpha=2.0 datamodule.train_weight_exponent=2.5 trainer.max_epochs=200
# CE+US
python run.py +experiment=celeba_undersampled loss_fn=cross_entropy
# VS
python run.py +experiment=celeba_erm loss_fn=vs_group_loss loss_fn.gamma=0.4 loss_fn.num_per_group="[1446,1308,468,33]"
# Poly+DRO
python run.py +experiment=celeba_dro loss_fn=polynomial_loss loss_fn.alpha=2.0 optimizer.lr=0.001 trainer.max_epochs=200 model.adv_probs_lr=0.05
# CE+DRO
python run.py +experiment=celeba_dro loss_fn=cross_entropy optimizer.lr=0.001 trainer.max_epochs=200 model.adv_probs_lr=0.05
# VS+DRO
python run.py +experiment=celeba_dro loss_fn=vs_group_loss loss_fn.gamma=0.4 loss_fn.num_per_group="[1446,1308,468,33]" optimizer.lr=0.001 trainer.max_epochs=200 model.adv_probs_lr=0.05
```

## BibTeX

```
@inproceedings{
wang2022is,
title={Is Importance Weighting Incompatible with Interpolating Classifiers?},
author={Ke Alexander Wang and Niladri Shekhar Chatterji and Saminul Haque and Tatsunori Hashimoto},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=uqBOne3LUKy}
}
```
