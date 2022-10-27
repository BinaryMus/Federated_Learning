# Federated_Learning

``````
from federated.core.trainers import Trainer

if __name__ == '__main__':
    trainer = Trainer(n_clients=10,
                      optimizer="SGD",
                      model="SimpleCNN",
                      data="MNIST",
                      lr=0.01,
                      batch_size=64,
                      path="./datasets",
                      alpha=100,
                      local_epoch=3,
                      global_epoch=5,
                      device="cpu")
    trainer.train()

``````