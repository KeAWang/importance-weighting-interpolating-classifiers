import wandb
import os
import pickle


def get_history(user="", project="", query={}, **kwargs):
    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=query)
    dataframes = [run.history(**kwargs) for run in runs]
    return list(zip(runs, dataframes))


def download_files(user="", project="", query={}, save_dir=".", **kwargs):
    """
    Download the files of each run into a new directory for the run.
    Also saves the config dict of the run.

    See https://docs.wandb.com/library/reference/wandb_api for how to write queries
    """
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=query)
    for run in runs:
        name = run.name
        config = run.config

        run_dir = os.path.join(save_dir, name)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)

        with open(os.path.join(run_dir, "config.pkl"), "wb") as h:
            pickle.dump(config, h)

        files = run.files()
        for file in files:
            file.download(root=run_dir)
    return


def get_config(user="", project="", query={}):
    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=query)

    configs = [(run.name, run.config) for run in runs]
    return configs


def config_to_omegaconf(config: dict):
    from omegaconf import OmegaConf

    keys, values = zip(*config.items())

    # convert from keys that look like "datamodules/batch_size" into "datamodules.batch_size"
    dot_keys = [key.replace("/", ".") for key in keys]

    # convert "None" strings into "null" for OmegaConf to parse it as a None object
    new_values = ["null" if v == "None" else v for v in values]

    dot_list = [f"{k}={v}" for k, v in zip(dot_keys, new_values)]
    omega_conf = OmegaConf.from_dotlist(dot_list)
    return omega_conf
