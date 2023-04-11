import argparse

from federated.core.configs import Config
from federated.core.utils import seed_it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./examples/FedAVG/config_mp.yml", type=str, help='config file')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--arch', default="mp", type=str)
    arg = parser.parse_args()
    seed_it(arg.seed)
    config = Config(arg.config)
    if arg.arch == "mp":
        config.run_mp()
    else:
        config.run_distributed()


if __name__ == '__main__':
    main()
