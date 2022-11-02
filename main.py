# import argparse
#
# from federated.core.configs import Config
# from federated.core.utils import seed_it
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', default="./config.yml", type=str, help='config file')
#     parser.add_argument('--seed', default=42, type=int)
#     arg = parser.parse_args()
#     seed_it(arg.seed)
#     config = Config(arg.config)
#     config.run()


if __name__ == '__main__':
    from federated.core.trainers.mp_trainer import Trainer

    cluster_conf = {
        "ps": ("localhost", 9000),
        "client": [
            ("localhost", i)
            for i in range(9001, 9005)
        ]
    }
    trainer = Trainer(
        cluster_conf,
        "SGD",
        "SimpleCNN",
        "MNIST",
        0.1,
        64,
        "./datasets",
        100,
        3,
        10)
    trainer.run()
