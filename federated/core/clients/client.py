import pickle
import socket
import time

import torch
from torch.utils.data import DataLoader


class BaseClient:
    def __init__(
            self,
            ip: str,
            port: int,
            server_ip: str,
            server_port: int,
            model: str,
            data: DataLoader,
            sample_num: int,
            n_classes: int,
            global_epoch: int,
            local_epoch: int,
            optimizer: str,
            lr: float,
            device: str,
            criterion=torch.nn.CrossEntropyLoss()
    ):
        self.model = None
        self.optimizer = None
        self.ip = ip
        self.port = port
        self.server_ip = server_ip
        self.server_port = server_port
        self.criterion = criterion  # 损失函数
        self.data = data  # 数据
        self.sample_num = sample_num
        self.device = torch.device(device)  # 设备
        self.lr = lr  # 学习率
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch  # 本地多轮迭代次数
        self.loss = []  # 本地训练的损失
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model_name = model
        self.optim_name = optimizer
        self.n_classes = n_classes
        print(f"CLIENT@{self.ip}:{self.port} INFO: Start!")

    def first_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        self.client_connect(client_socket, self.server_ip, self.server_port, self.ip, self.port)
        self.model.load_state_dict(self.client_recv(client_socket))

        client_socket.close()

    def run(self):
        from ..register import all_arch, all_optim
        self.model = all_arch[self.model_name](num_classes=self.n_classes).to(self.device)  # 模型
        self.optimizer = all_optim[self.optim_name](self.model.parameters(), lr=self.lr)
        self.first_pull()
        for ge in range(self.global_epoch):
            for epoch in range(self.local_epoch):
                loss_avg = self.train()
                self.loss.append(loss_avg)
                print(
                    f"CLIENT@{self.ip}:{self.port} INFO: "
                    f"Global Epoch[{ge + 1}|{self.global_epoch}] "
                    f"Local Epoch[{epoch + 1}|{self.local_epoch}] "
                    f"Loss:{round(loss_avg, 3)}")
            self.push_pull()

    def train(self):
        loss_avg = 0
        cnt = 0
        for x, y in self.data:
            cnt += 1
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            loss_avg += loss.item()
            self.optimizer.step()
        return loss_avg / cnt

    def push_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        self.client_connect(client_socket, self.server_ip, self.server_port, self.ip, self.port)
        client_socket.sendall(pickle.dumps([self.sample_num, self.model.state_dict()]))
        client_socket.sendall(b'stop!')

        self.model.load_state_dict(self.client_recv(client_socket))

        client_socket.close()

    @staticmethod
    def client_recv(client_socket):
        new_para = b''
        tmp = client_socket.recv(1024)
        while tmp:
            if tmp.endswith(b'stop!'):
                new_para += tmp[:-5]
                break
            new_para += tmp
            tmp = client_socket.recv(1024)
        return pickle.loads(new_para)

    @staticmethod
    def client_connect(client_socket, server_ip, server_port, ip, port):
        while True:
            try:
                client_socket.connect((server_ip, server_port))
                break
            except Exception as e:
                print(f"CLIENT@{ip}:{port} ERROR: {e}, reconnecting!")
                time.sleep(1)
