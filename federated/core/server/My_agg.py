import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from . import server
import copy


class My(server.BaseServer):
    clients_weight = [0 for _ in range(5)]
    def __init__(self, epoch: int, clients: List, model: torch.nn.Module, data: DataLoader, device: str, alpha: float = 0):
        super().__init__(epoch, clients, model, data, device)
        self.para_cache = []
        self.alpha = alpha
        self.beta = [0 for _ in range(self.n_clients)]
        self.round_weight = [0 for _ in range(self.n_clients)]
        self.round_final_weight = [0 for _ in range(self.n_clients)]
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
        norm = 0
        _1dvector = torch.zeros(self.dim)
        for key in model:
            tmp = model[key].view(-1)
            for x in tmp:
                _1dvector[idx] += x
                idx += 1
                norm += x*x
        return _1dvector,norm**0.5

    def my(self,p: float = 2):
        agg_para_cache = copy.deepcopy(self.clients[0].model)
        clear_parameter(agg_para_cache)
        server_model,self.clients_norm[0] = self.to_1dvector(self.para_cache[0])
        for i in range(1,self.n_clients):
            tmp_model,self.clients_norm[i] = self.to_1dvector(self.para_cache[i])
            self.round_weight[i] = 1.0/((self.Lp_distance(server_model,tmp_model)+self.alpha)**p)
        round_sum = sum(self.round_weight)
        for i in range(1,self.n_clients):
            self.beta[i] = min(self.round_weight[i]*(self.n_clients-1)/round_sum,1)
        for i in range(1,self.n_clients):
            self.clients_weight[i] = (self.beta[i]*self.clients_weight[i] + self.round_weight[i])/2
        clients_sum = sum(self.clients_weight)
        for i in range(1,self.n_clients):
            self.round_final_weight[i] = self.clients_weight[i]/clients_sum
        for i in range(1,self.n_clients):
            for key in self.model.state_dict():
                self.model.state_dict()[key] += self.round_final_weight[i] * self.para_cache[i][key]

    def pull(self, client_nums, total):
        clear_parameter(self.model)
        self.my()