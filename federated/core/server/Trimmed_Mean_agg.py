import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from . import server


class Trimmed_Mean(server.BaseServer):
    def __init__(self, epoch: int, clients: List, model: torch.nn.Module, data: DataLoader, device: str, f: int = 8):
        super().__init__(epoch, clients, model, data, device)
        self.para_cache = []
        self.f = min(f,(self.n_clients-1)//2)
        for i in range(self.n_clients):
            self.para_cache.append(clients[i].model.state_dict())
        self.dim =  0
        for key in self.model.state_dict():
            self.dim += self.model.state_dict()[key].view(-1).size(0)
        print(self.dim)

    def trimmed_mean(self):
        for key in self.para_cache[0]:
            tensor_shape = self.para_cache[0][key].shape
            tensor_size = self.para_cache[0][key].view(-1).size(0)
            tensor_mean = torch.zeros(tensor_size)
            tensor_all_clients = [[] for _ in range(tensor_size)]
            for i in range(self.n_clients):
                m = self.para_cache[i][key].view(-1)
                for x in range(tensor_size):
                    tensor_all_clients[x].append(m[x])
            for i in range(tensor_size):
                tensor_mean[i] = sum(tensor_all_clients[i][self.f:self.n_clients-self.f])/(self.n_clients-2*self.f)
            dtype = self.model.state_dict()[key].dtype
            self.model.state_dict()[key] += tensor_mean.view(tensor_shape).to(dtype)
    def pull(self, client_nums, total):
        # clear_parameter(self.model)
        self.trimmed_mean()
        #print(self.model.state_dict())