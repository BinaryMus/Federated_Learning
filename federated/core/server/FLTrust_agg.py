import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from .. import server
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
            
        self.dim = sum(p.numel() for p in self.model.parameters())
        print(self.dim)

    def to_1dvector(self,model):
        _1dvector = torch.cat([value.view(-1) for value in model.values()])
        norm = torch.norm(_1dvector, p=2)
        return _1dvector,norm


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
                    dtype = self.clients[0].model.state_dict()[key].dtype
                    agg_para_cache.state_dict()[key] += (cos_theta*self.clients_norm[0]/self.clients_norm[i] * self.para_cache[i][key]).to(dtype)
        normalized = 1/sum(self.clients_weight)
        for key in self.model.state_dict():
            dtype = self.model.state_dict()[key].dtype
            self.model.state_dict()[key] += (normalized * agg_para_cache.state_dict()[key]).to(dtype)

    def pull(self, client_nums, total):
        # for i in range(self.n_clients):
        #     for key in self.para_cache[0]:
        #         self.para_cache[i][key] -= self.model.state_dict()[key]
        # clear_parameter(self.model)
        self.fltrust()