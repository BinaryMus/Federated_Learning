from .. import Server, Client
from ...models import *
from ...datasets import *


class Trainer:
    def __init__(self,
                 n_clients: int,
                 optimizer: str,
                 model: str,
                 data: str,
                 lr: float,
                 batch_size: int,
                 path: str,
                 alpha: float,
                 local_epoch: int,
                 global_epoch: int,
                 device: str = "cuda",
                 ):
        self.n_clients = n_clients
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.path = path
        self.alpha = alpha
        self.local_epoch = local_epoch
        self.global_epoch = global_epoch
        self.device = device
        self.acc1_lst = []
        self.acc5_lst = []
        if model == "SimpleCNN" and data == "MNIST" and optimizer == "SGD":
            data = Mnist(self.n_clients,
                         self.batch_size,
                         self.path,
                         self.alpha,
                         )
            self.clients = [
                Client(i,
                       self.n_clients,
                       SimpleCNN(),
                       data.trainLoader[i],
                       local_epoch,
                       self.optimizer,
                       self.lr,
                       self.device, ) for i in range(self.n_clients)]
            self.server = Server(global_epoch,
                                 self.clients,
                                 SimpleCNN(),
                                 data.validationLoader,
                                 self.device)
        else:
            assert True, "暂不支持"

    def train(self):
        for epoch in range(self.global_epoch):
            for idx in range(self.n_clients):
                self.clients[idx].train_loop()
            self.server.pull()
            self.server.push()
            acc1, acc5 = self.server.validate()
            self.acc1_lst.append(acc1)
            self.acc5_lst.append(acc5)
            print(f"Global_Epoch[{epoch + 1}|{self.global_epoch}] Top-1_Accuracy: {acc1} Top-5_Accuracy: {acc5}")
