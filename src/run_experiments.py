import argparse
from pathlib import Path

from train import load_config, run_training


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = [
    "baseline",
    "augmentation",
    "batchnorm",
    "dropout",
    "label_smoothing",
    "all_tricks",
]


def resolve_config_path(config_name):
    path = Path(config_name)

    if path.suffix == ".yaml":
        return path if path.is_absolute() else REPO_ROOT / path

    return REPO_ROOT / "configs" / f"{config_name}.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 ablation experiments sequentially."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Temporarily override the number of epochs for this run."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Temporarily override the random seed for this run."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Config names or YAML paths to run, in order."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for config_name in args.configs:
        config_path = resolve_config_path(config_name)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config = load_config(config_path)

        if args.epochs is not None:
            config["epochs"] = args.epochs

        if args.seed is not None:
            config["seed"] = args.seed

        print(f"\n=== Running {config['experiment_name']} ===")
        run_training(config)


if __name__ == "__main__":
    main()
