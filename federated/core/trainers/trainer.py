from ..server.Krum_agg import Krum
from ..server.Multi_Krum_agg import Multi_Krum
from ..server.Median_agg import Median
from ..server.Trimmed_Mean_agg import Trimmed_Mean
from ..server.FLTrust_agg import FLTrust
from ..server.My_agg import My
from ..server.CC_agg import CC
from ..server.NBH_agg import NBH
from ..server.CBH_agg import CBH
from ..server.MyM_agg import MyM
from ..server.MyA_agg import MyA
from ..server.MyS_agg import MyS
from .. import BaseClient, BaseServer
from ...models import *
from ...datasets import *
from ..utils import clear_parameter

all_arch = {"SimpleCNN": SimpleCNN, "VGG11": VGG11, "ResNet18": Resnet18}
all_data = {"MNIST": Mnist, "CIFAR10": Cifar10}
all_server = {"FedAVG": BaseServer, "Krum": Krum, "Multi_Krum": Multi_Krum, "Median": Median, "Trimmed_Mean": Trimmed_Mean, "FLTrust": FLTrust, "My": My, "CC": CC, "LFH": CC, "CBH": CBH, "NBH": NBH, "NB": NBH, "MyM":MyM, "MyA": MyA, "MyS": MyS}
all_client = {"FedAVG": BaseClient, "Krum": BaseClient, "Multi_Krum": BaseClient, "Median": BaseClient, "Trimmed_Mean": BaseClient, "FLTrust": BaseClient, "My": BaseClient, "CC": BaseClient, "LFH": BaseClient, "CBH": BaseClient, "NBH": BaseClient, "NB": BaseClient, "MyM":BaseClient, "MyA": BaseClient, "MyS": BaseClient}


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
                 atk: int,
                 frac: float,
                 filefolder: str,
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
        self.atk = atk
        self.frac = frac
        self.filefolder = filefolder

        self.data = all_data[data](self.n_clients,
                                   self.batch_size,
                                   self.path,
                                   self.alpha,
                                   flipfrac = 0
                                   )
        
        self.momentum = [all_arch[model](num_classes=len(self.data.train_set.classes), ) for _ in range(n_clients)]
        for i in range(n_clients):
            clear_parameter(self.momentum[i])
        self.His_beta = 1 - 0.9

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
        folders = "experiment/" + self.filefolder
        # folders = "experiment/20c_reverse/0.4_iid_100epoch_lr=0.01/"
        # folders = "experiment/20c_nonbyz/20c_iid_500epoch_lr=0.01/"
        attack_info = ""
        f =  open(folders + (str)(self.algorithm) + "_" + (str)(self.global_epoch) + attack_info + ".txt","w")
        self.server.push()
        for epoch in range(self.global_epoch):
            for idx in range(self.n_clients):
                self.clients[idx].train_loop()

            if(self.algorithm == "LFH" or self.algorithm == "CBH" or self.algorithm == "NBH"):
                for i in range(self.n_clients):
                    for key in self.clients[0].model.state_dict():
                        self.momentum[i].state_dict()[key] += self.His_beta * (self.clients[i].model.state_dict()[key] - self.server.model.state_dict()[key] - self.momentum[i].state_dict()[key])
                        # self.clients[i].model.state_dict()[key] -=  self.His_beta * (self.clients[i].model.state_dict()[key] - self.server.model.state_dict()[key])
                        self.clients[i].model.state_dict()[key] += self.momentum[i].state_dict()[key] - self.clients[i].model.state_dict()[key] + self.server.model.state_dict()[key]

            for i in range(self.n_clients):
                for key in self.clients[0].model.state_dict():
                    self.clients[i].model.state_dict()[key] -= self.server.model.state_dict()[key]

            self.server.attack(self.atk, self.frac)

            acc1, acc5 = self.server.pull_push(self.data.client_nums, self.data.total)
            self.acc1_lst.append(acc1)
            self.acc5_lst.append(acc5)
            f.write(f"{acc1}\n")
            print(
                f"SERVER INFO: Global_Epoch[{epoch + 1}|{self.global_epoch}] "
                f"Top-1_Accuracy: {acc1} Top-5_Accuracy: {acc5}")
