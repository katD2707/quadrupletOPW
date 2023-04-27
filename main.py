from trainer import Trainer
from argparse import ArgumentParser
import yaml
import utils


def main(args):
    trainer = Trainer(args.train_args, args.model_args)

    trainer.train()


if __name__ == "__main__":
    # Parse parameters
    parser = ArgumentParser(description="train the TitaNet model")
    parser.add_argument(
        "-p",
        "--params",
        help="path for the parameters .yml file",
        required=True,
        type=str,
    )
    args = parser.parse_args()
    with open(args.params, "r") as params:
        args = yaml.load(params, Loader=yaml.FullLoader)
    params = utils.Struct(**args)

    main(params)
