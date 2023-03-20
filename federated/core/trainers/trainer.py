from ..server.Krum_agg import Krum
from ..server.Multi_Krum_agg import Multi_Krum
from .. import BaseClient, BaseServer
from ...models import *
from ...datasets import *

all_arch = {"SimpleCNN": SimpleCNN, "VGG11": VGG11, "ResNet18": Resnet18}
all_data = {"MNIST": Mnist, "CIFAR10": Cifar10}
all_server = {"FedAVG": BaseServer, "Krum": Krum, "Multi_Krum": Multi_Krum}
all_client = {"FedAVG": BaseClient, "Krum": BaseClient, "Multi_Krum": BaseClient}


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
                 algorithm: str = "FedAVG",
                 ):
        self.n_clients = n_clients  # 客户端数量
        self.model = model  # 模型架构
        self.data_name = data  # 数据
        self.optimizer = optimizer  # 优化器
        self.lr = lr  # 学习率
        self.batch_size = batch_size  # mini-batch大小
        self.path = path  # 数据集路径
        self.alpha = alpha  # 数据NON-IID程度
        self.local_epoch = local_epoch  # 本地多轮迭代次数
        self.global_epoch = global_epoch  # 全局迭代次数
        self.device = device  # 使用设备
        self.algorithm = algorithm  # 算法
        self.acc1_lst = []
        self.acc5_lst = []

        self.data = all_data[data](self.n_clients,
                                   self.batch_size,
                                   self.path,
                                   self.alpha,
                                   )
        self.clients = [all_client[algorithm]
                        (i,
                         self.n_clients,
                         all_arch[model](num_classes=len(self.data.train_set.classes), ),
                         self.data.trainLoader[i],
                         local_epoch,
                         self.optimizer,
                         self.lr,
                         self.device, ) for i in range(self.n_clients)]
        self.server = all_server[algorithm](global_epoch,
                                            self.clients,
                                            all_arch[model](num_classes=len(self.data.train_set.classes), ),
                                            self.data.validationLoader,
                                            self.device)
        print(f"TRAINER INFO: "
              f"\n\tclient number: {n_clients}"
              f"\n\tmodel: {model}"
              f"\n\toptimizer: {optimizer}, learning_rate: {lr}"
              f"\n\tdata: {data} data_path: {path}, alpha: {alpha}, batch_size: {batch_size}"
              f"\n\tlocal_epoch: {local_epoch}, global_epoch: {global_epoch}"
              f"\n\talgorithm: {algorithm}"
              f"\n\tdevice: {device}")

    def train(self):
        print("TRAINER INFO: Start Training!")
        self.server.push()
        for epoch in range(self.global_epoch):
            for idx in range(self.n_clients):
                self.clients[idx].train_loop()

            acc1, acc5 = self.server.pull_push(self.data.client_nums, self.data.total)
            self.acc1_lst.append(acc1)
            self.acc5_lst.append(acc5)
            print(
                f"SERVER INFO: Global_Epoch[{epoch + 1}|{self.global_epoch}] "
                f"Top-1_Accuracy: {acc1} Top-5_Accuracy: {acc5}")
