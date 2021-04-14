import wandb
import os
import pickle


def get_history(user="", project="", query={},
                **kwargs):
    api = wandb.Api()
    runs = api.runs(path=f"{user}/{project}", filters=query)
    dataframes = [run.history(**kwargs) for run in runs]
    return list(zip(runs, dataframes))


def download_files(user="", project="",
                   query={}, save_dir=".", **kwargs):
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