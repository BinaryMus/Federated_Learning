import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from . import server
import copy


class FLTrust(server.BaseServer):
    def __init__(self, epoch: int, clients: List, model: torch.nn.Module, data: DataLoader, device: str):
        super().__init__(epoch, clients, model, data, device)
        self.para_cache = []
        # 2-norm of clients model
        self.clients_norm = [0 for _ in range(self.n_clients)]
        # clients truth score
        self.clients_weight = [0 for _ in range(self.n_clients)]
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

    def fltrust(self):
        agg_para_cache = copy.deepcopy(self.clients[0].model)
        clear_parameter(agg_para_cache)
        server_model,self.clients_norm[0] = self.to_1dvector(self.para_cache[0])
        for i in range(1,self.n_clients):
            tmp_model,self.clients_norm[i] = self.to_1dvector(self.para_cache[i])
            cos_theta = torch.dot(server_model,tmp_model).item()/(self.clients_norm[0]*self.clients_norm[i])
            self.clients_weight[i] = cos_theta if cos_theta > 0 else 0
            if cos_theta > 0:
                for key in self.para_cache[i]:
                    agg_para_cache.state_dict()[key] += cos_theta*self.clients_norm[0]/self.clients_norm[i] * self.para_cache[i][key]
        normalized = 1/sum(self.clients_weight)
        for key in self.model.state_dict():
            self.model.state_dict()[key] += normalized * agg_para_cache.state_dict()[key]

    def pull(self, client_nums, total):
        clear_parameter(self.model)
        self.fltrust()