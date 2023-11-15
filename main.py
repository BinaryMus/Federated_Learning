import argparse

from federated.core.configs import Config
from federated.core.utils import seed_it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./config_mp.yml", type=str, help='config file')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--algo', default='FedAVG', type=str)
    parser.add_argument('--atk', default=0, type=int)
    parser.add_argument('--frac', default=0.0, type=float)
    parser.add_argument('--filefolder', default='error/', type=str)
    
    arg = parser.parse_args()
    seed_it(arg.seed)
    config = Config(arg.config)
    config.config['algorithm'] = arg.algo
    config.run()


if __name__ == '__main__':
    main()
