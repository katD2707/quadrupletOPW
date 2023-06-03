from train.trainer import Trainer
from argparse import ArgumentParser
import yaml
import collections
import utils
import model as module_arch
import data as module_data
import train as module_train
from configs import ConfigParser


def main(args):
    model = config.init_obj("model_args", module_arch).cuda()

    dataset = config.init_obj("data_args", module_data)

    training_args = config.init_obj("training_args", module_train)

    # QOP(
    #     dim=model_args.dim,
    #     std=model_args.std,
    #     lambda_2=model_args.lambda_2,
    #     lambda_3=model_args.lambda_3,
    #     verbose=model_args.verbose,
    #     p_norm=model_args.p_norm,
    #     sinkhorn_maxIter=model_args.sinkhorn_maxIter,
    #     tol=model_args.tol,
    #     alpha=model_args.alpha,
    #     beta=model_args.beta,
    # ).cuda()

    trainer = Trainer(dataset=dataset, train_args=training_args, model=model)

    trainer.train()


if __name__ == "__main__":
    # Parse parameters
    args = ArgumentParser(description="train the TitaNet model")
    args.add_argument(
        "-c",
        "--config",
        help="path for the parameters .yml file",
        required=True,
        type=str,
    )

    args.add_argument(
        "-d",
        "--device",
        default="cuda",
        type=str,
        help="type of device",
    )

    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
        CustomArgs(["--ep", "--epochs"], type=int, target="trainer;epochs"),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
