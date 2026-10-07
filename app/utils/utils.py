import logging
import sys
from logging import Logger
from types import SimpleNamespace

import torch
import yaml


def instantiate_logger(logger_name: str) -> Logger:
    """Instantiate Logger object.

    Parameters
    ----------
    logger_name : str
        Name of logger.

    Returns
    -------
    Logger
        Logger object.
    """

    # Set logger config
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def to_namespace(data):
    if isinstance(data, dict):
        # Recursively convert values, then wrap the dict in a SimpleNamespace 
        return SimpleNamespace(**{k: to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        # Recursively convert items inside lists
        return [to_namespace(item) for item in data]
    else:
        # Return primitive types (strings, ints, etc.) as-is
        return data
    
def load_config(config_path: str, split: str) -> SimpleNamespace:
    """Load config. 

    Parameters
    ----------
    config_path : str
        Config file location.
    split : str
        Choose one of ["train", "test", "predict"].
    Returns
    -------
    SimpleNamespace
        Dictionary like object for easy access config's key value.
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)[split]

    config =  to_namespace(config)

    return config

# set get device function
def get_device() -> torch.device:
    """Get machine device, priority cuda >> mps >> cpu.

    Returns
    -------
    torch.device
        Device for run the model.
    """
    if torch.cuda.is_available():
        device_name = "cuda"
    elif torch.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"

    return torch.device(device_name)

