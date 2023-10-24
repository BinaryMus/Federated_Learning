import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from . import server
import copy


class CC(server.BaseServer):
    def __init__(self, epoch: int, clients: List, model: torch.nn.Module, data: DataLoader, device: str,tau: float = 0.1):
        super().__init__(epoch, clients, model, data, device)
        self.tau = tau
        self.para_cache = []
        for i in range(self.n_clients):
            self.para_cache.append(clients[i].model.state_dict())
        self.dim =  0
        for key in self.model.state_dict():
            self.dim += self.model.state_dict()[key].view(-1).size(0)
        print(self.dim)

    def to_1dvector(self,model):
        idx = 0
        norm = 0
        _1dvector = torch.zeros(self.dim)
        for key in model:
            tmp = model[key].view(-1)
            for x in tmp:
                _1dvector[idx] += x
                idx += 1
                norm += x*x
        return _1dvector,norm**0.5

    def centered_clipping(self):
        for i in range(self.n_clients):
            tmp,weight = self.to_1dvector(self.para_cache[i])
            weight = min(self.tau/weight,1)
            for key in self.model.state_dict():
                self.model.state_dict()[key] += (1.0/self.n_clients) * (weight * self.para_cache[i][key])

    def pull(self, client_nums, total):
        # for i in range(self.n_clients):
        #     for key in self.para_cache[0]:
        #         self.para_cache[i][key] -= self.model.state_dict()[key]
        # clear_parameter(self.model)
        self.centered_clipping()