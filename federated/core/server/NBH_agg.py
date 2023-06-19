import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from . import server
import copy
import math


class NBH(server.BaseServer):
    def __init__(self, epoch: int, clients: List, model: torch.nn.Module, data: DataLoader, device: str, alpha: float = 1.5):
        super().__init__(epoch, clients, model, data, device)
        self.para_cache = []
        self.alpha = alpha
        self.clients_norm = [0 for _ in range(self.n_clients)]
        for i in range(self.n_clients):
            self.para_cache.append(clients[i].model.state_dict())
        self.dim =  0
        for key in self.model.state_dict():
            self.dim += self.model.state_dict()[key].view(-1).size(0)
        print(self.dim)
    
    def Lp_distance(self,model_a,model_b,p: float = 2):
        dis = 0
        for x,y in zip(model_a,model_b):
            dis += (x.item()-y.item())**p
        return dis**(1.0/p)

    def to_1dvector(self,model):
        idx = 0
        norm = 0.0
        _1dvector = torch.zeros(self.dim)
        for key in model:
            tmp = model[key].view(-1)
            for x in tmp:
                _1dvector[idx] += x
                idx += 1
                norm += x*x
        return _1dvector,norm**0.5
    
    def norm_p(self,model,p: float = 2):
        norm = 0.0
        for x in model:
            norm += x*x
        return norm ** (1.0/p)

    def nbh(self):
        agg_para_cache = copy.deepcopy(self.clients[0].model)
        clear_parameter(agg_para_cache)
        server_model,self.clients_norm[0] = self.to_1dvector(self.para_cache[0])
        cnt = 1.0
        print(f'server:{self.clients_norm[0]}\n')
        for i in range(1,self.n_clients):
            tmp_model, _tmp_norm = self.to_1dvector(self.para_cache[i])
            self.clients_norm[i] = self.norm_p(tmp_model - server_model)
            print(f'client{i}:{self.clients_norm[i]}\n')
            if self.clients_norm[i] <= self.clients_norm[0] * self.alpha:
                cnt += 1
        for i in range(0,self.n_clients):
            if self.clients_norm[i] <= self.clients_norm[0] * self.alpha or i == 0:
                print(f'access:{i}\n')
                for key in self.model.state_dict():
                    self.model.state_dict()[key] += (1.0/cnt) * self.para_cache[i][key]

    def pull(self, client_nums, total):
        # for i in range(self.n_clients):
        #     for key in self.para_cache[0]:
        #         self.para_cache[i][key] -= self.model.state_dict()[key]
        # clear_parameter(self.model)
        self.nbh()