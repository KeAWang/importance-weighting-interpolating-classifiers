import dotenv
import hydra
from omegaconf import DictConfig


def set_personal_dir_from_hostname():
    from pathlib import Path
    import platform

    import os
    import pwd

    username = pwd.getpwuid(os.getuid())[0]

    machine_name = platform.node().split(".")[0]
    local_dir = Path(f"/{machine_name}")
    for scr in ["scr0", "scr1", "scr", "scr2"]:
        personal_dir = local_dir / scr / username
        if personal_dir.exists():
            break
    os.environ["PERSONAL_DIR"] = str(personal_dir)
    return


set_personal_dir_from_hostname()
# load environment variables from `.env` file if it exists
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import template_utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - disabling lightning logs
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    template_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        template_utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
