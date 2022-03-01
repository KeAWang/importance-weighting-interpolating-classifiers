This repo is created from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
Experiment settings are managed by [hydra](https://hydra.cc/), a hierarchical config framework, and the setting files are specified in the `configs/` directory. 

## Setup instructions

0. Make a weights and bias account
1. `conda env create -f conda_env.yml`
2. `pip install requirements.txt`
3. Copy `.env.tmp` to a new file called `.env`. Edit the `PERSONAL_DIR` environment variable in `.env` to be the root directory of where you want to store your data. 
4. Simplest command: `python run.py +experiment=celeba_erm`. 

## Reproducing experiments

TODO
