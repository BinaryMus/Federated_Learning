import argparse

from federated.core.configs import Config
from federated.core.utils import seed_it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config.yml", type=str, help='config file')
    parser.add_argument('--seed', default=42, type=int)
    arg = parser.parse_args()
    seed_it(arg.seed)
    config = Config(arg.config)
    if config.config["arch"] == "mp":
        config.run_mp()
    elif config.config["arch"] == "distributed":
        config.run_distributed()


if __name__ == '__main__':
    main()
