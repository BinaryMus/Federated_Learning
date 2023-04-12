import socket
import pickle
import struct

import torch
from torch.utils.data import DataLoader

from . import BaseClient


class FedProxClient(BaseClient):
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
            criterion=torch.nn.CrossEntropyLoss(),
            MU: float = 0.05,
    ):
        super(FedProxClient, self).__init__(
            ip,
            port,
            server_ip,
            server_port,
            model,
            data,
            sample_num,
            n_classes,
            global_epoch,
            local_epoch,
            optimizer,
            lr,
            device,
            criterion)
        self.model_parameter = None
        self.MU = MU

    def train(self):
        loss_avg = 0
        cnt = 0
        for x, y in self.data:
            cnt += 1
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            proximal_term = 0.0
            for w, w_t in zip(self.model_parameter, self.model.parameters()):
                proximal_term += (w - w_t).norm(2)
            loss = self.criterion(output, y) + (self.MU / 2) * proximal_term
            loss.backward()
            loss_avg += loss.item()
            self.optimizer.step()
        return loss_avg / cnt

    def push_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))
        client_socket.sendall(pickle.dumps([self.sample_num, self.model.state_dict()]))
        client_socket.sendall(b'stop!')

        self.model.load_state_dict(self.client_recv(client_socket))
        self.model_parameter = self.model.parameters()

        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client_socket.close()

    def first_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))

        self.model.load_state_dict(self.client_recv(client_socket))
        self.model_parameter = self.model.parameters()

        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client_socket.close()
