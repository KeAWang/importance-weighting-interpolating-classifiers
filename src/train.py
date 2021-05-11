import logging
from typing import List, Optional
import torch

from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning import seed_everything

import hydra
from omegaconf import DictConfig

from dataclasses import dataclass
from src.utils import template_utils

log = logging.getLogger(__name__)


@dataclass
class HydraObjects:
    """Class for keeping track of objectes to initialize using hydra"""

    datamodule: LightningDataModule
    architecture: torch.nn.Module
    model: LightningModule
    callbacks: List[Callback]
    logger: List[LightningLoggerBase]
    trainer: Trainer


def hydra_init(config: DictConfig, train=True) -> HydraObjects:
    """ Initialize the objects from a hydra config.

    We log only when `train` is True.

    """
    # Init Lightning datamodule
    if train:
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    input_size = datamodule.size()
    output_size = datamodule.num_classes
    # Init function approximator
    architecture: torch.nn.Module = hydra.utils.instantiate(
        config.architecture, input_size=input_size, output_size=output_size
    )

    # Compute class weights if required
    # TODO: allow for group_weights
    class_weights = (
        torch.tensor(config.model["class_weights"], dtype=torch.get_default_dtype())
        if config.model.get("class_weights")
        else torch.ones((output_size,), dtype=torch.get_default_dtype())
    )

    # Init Lightning model
    if train:
        log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        config.model,
        architecture=architecture,
        class_weights=class_weights,
        optimizer=config.optimizer,
        loss_fn=config.loss_fn,
        output_size=output_size,
        _recursive_=False,
    )

    if train:
        # Init Lightning callbacks
        callbacks: List[Callback] = []
        if "callbacks" in config:
            for _, cb_conf in config["callbacks"].items():
                if "_target_" in cb_conf:
                    log.info(f"Instantiating callback <{cb_conf._target_}>")
                    callbacks.append(hydra.utils.instantiate(cb_conf))
    else:
        callbacks = []

    # Init Lightning loggers
    if train:
        logger: List[LightningLoggerBase] = []
        if "logger" in config:
            for _, lg_conf in config["logger"].items():
                if type(lg_conf) is DictConfig and "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    logger.append(hydra.utils.instantiate(lg_conf))
    else:
        logger = []

    # Init Lightning trainer
    if train:
        log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    return HydraObjects(
        datamodule=datamodule,
        architecture=architecture,
        model=model,
        callbacks=callbacks,
        logger=logger,
        trainer=trainer,
    )


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed)

    hydra_objects = hydra_init(config)
    model = hydra_objects.model
    datamodule = hydra_objects.datamodule
    trainer = hydra_objects.trainer
    callbacks = hydra_objects.callbacks
    logger = hydra_objects.logger

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    template_utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    # if not config.trainer.get("fast_dev_run"):
    #    log.info("Starting testing!")
    #    trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    template_utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    # log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for Optuna optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
