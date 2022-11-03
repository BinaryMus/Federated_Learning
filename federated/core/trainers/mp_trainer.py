import multiprocessing as mp

from .. import BaseClient, BaseServer
from ...datasets import *

# all_arch = {"SimpleCNN": SimpleCNN, "VGG11": VGG11, "ResNet18": Resnet18}
all_data = {"MNIST": Mnist, "CIFAR10": Cifar10}
all_server = {"FedAVG": BaseServer}
all_client = {"FedAVG": BaseClient}


class TrainerMP:
    def __init__(
            self,
            cluster_conf: dict,
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
        self.cluster_conf = cluster_conf  # 集群信息
        self.server_ip = self.cluster_conf['ps'][0]
        self.server_port = self.cluster_conf['ps'][1]
        self.clients_ip_port = []
        for client in self.cluster_conf['client']:
            self.clients_ip_port.append(client)

        self.n_clients = len(self.clients_ip_port)  # 客户端数量
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

        self.data = all_data[data](
            self.n_clients,
            self.batch_size,
            self.path,
            self.alpha,
        )

        self.clients = [all_client[algorithm](
            self.clients_ip_port[i][0],
            self.clients_ip_port[i][1],
            self.server_ip,
            self.server_port,
            self.model,
            # all_arch[model](num_classes=len(self.data.train_set.classes), ),
            self.data.trainLoader[i],
            len(self.data.train_set),
            len(self.data.train_set.classes),
            global_epoch,
            local_epoch,
            self.optimizer,
            self.lr,
            self.device, )
            for i in range(self.n_clients)]

        self.server = all_server[algorithm](
            self.server_ip,
            self.server_port,
            global_epoch,
            self.n_clients,
            self.model,
            # all_arch[model](num_classes=len(self.data.train_set.classes), ),
            self.data.validationLoader,
            len(self.data.train_set.classes),
            self.device)

        self.server_process = mp.Process(
            target=self.server.run,
            name=f"SERVER@{self.server_ip}:{self.server_port}")

        self.clients_process = [mp.Process(
            target=self.clients[i].run,
            name=f"CLIENT@{self.clients_ip_port[i][0]}:{self.clients_ip_port[i][0]}")
            for i in range(self.n_clients)]

        print(
            f"TRAINER INFO: "
            f"\n\tclient number: {self.n_clients}"
            f"\n\tmodel: {model}"
            f"\n\toptimizer: {optimizer}, learning_rate: {lr}"
            f"\n\tdata: {data} data_path: {path}, alpha: {alpha}, batch_size: {batch_size}"
            f"\n\tlocal_epoch: {local_epoch}, global_epoch: {global_epoch}"
            f"\n\talgorithm: {algorithm}"
            f"\n\tdevice: {device}")

    def run(self):
        self.server_process.start()
        for c in self.clients_process:
            c.start()

        self.server_process.join()
        for c in self.clients_process:
            c.join()
