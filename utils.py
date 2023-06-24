import os
from pathlib import Path
import numpy as np
import logging
import yaml
import torch


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


class KNN:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def forward(self, x, source_labels, labels):
        # labels: [num_subject*class, 1]
        topk = torch.topk(x, dim=-1, largest=False, k=self.n_neighbors)
        top_min_dist = topk.values
        matched_indices = topk.indices  # [B, K]

        labels = labels.T  # 1, num_train_samples
        labels = labels.expand_as(torch.empty(x.shape[0], labels.shape[1]))  # [[1,1,1,2,2,2]
        #  [1,1,1,2,2,2]
        #  [1,1,1,2,2,2]]
        count = 0
        for i in range(x.shape[0]):
            target_labels = labels[i][matched_indices[i]]
            final_labels = target_labels.mode().values

            count += sum(source_labels[i] == final_labels)
        return count



