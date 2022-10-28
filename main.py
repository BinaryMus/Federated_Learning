from federated.core.trainers import Trainer
from federated.core.utils import seed_it

if __name__ == '__main__':
    seed_it(42)
    trainer = Trainer(n_clients=2,
                      optimizer="SGD",
                      model="VGG11",
                      data="CIFAR10",
                      lr=0.01,
                      batch_size=64,
                      path="./datasets",
                      alpha=100,
                      local_epoch=3,
                      global_epoch=5,
                      device="cuda")
    trainer.train()
