import os
from pathlib import Path
import numpy as np
import logging
import yaml


def setup_logging(
    save_dir, log_config="utils/log/logger_config.yaml", default_level=logging.INFO
):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)

    if log_config.is_file():
        config = load_yaml(log_config)

        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        # Print config model
        # logging.config.dictConfig(config)

    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)


def load_yaml(fname):
    fname = Path(fname)
    with fname.open("rt") as file:
        config = yaml.safe_load(file)
    return config


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        yaml.dump(content, handle, indent=4, sort_keys=False)





