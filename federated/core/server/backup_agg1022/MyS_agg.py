import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from . import server
import copy
import math
import random as rd

# 基于欧氏距离和信誉的容错方法
# 信誉采样

class MyS(server.BaseServer):
    clients_weight = [[0 for _ in range(200)] for __ in range(20)]
    mu = 0.8
    select_frac = 0.2
    idx = 0
    def __init__(self, epoch: int, clients: List, model: torch.nn.Module, data: DataLoader, device: str):
        super().__init__(epoch, clients, model, data, device)
        self.para_cache = []
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

    def mys(self,p: float = 2):
        agg_para_cache = copy.deepcopy(self.clients[0].model)
        clear_parameter(agg_para_cache)
        server_model,self.clients_norm[0] = self.to_1dvector(self.para_cache[0])
        for i in range(1,self.n_clients):
            tmp_model,self.clients_norm[i] = self.to_1dvector(self.para_cache[i])
            # print(f"{i} dis:{self.Lp_distance(server_model,tmp_model)}\n")
            self.round_weight[i] = 1.0/((self.Lp_distance(server_model,tmp_model))**p)
            if math.isnan(self.round_weight[i]):
                self.round_weight[i] = 0
        round_sum = sum(self.round_weight)
        # print(f"round_sum:{round_sum}\n")

        select_set = []
        for i in range(1,self.n_clients):
            self.beta[i] = self.round_weight[i]/round_sum
            self.clients_weight[i][self.idx] = self.beta[i]
            if self.beta[i] > self.mu/(self.n_clients-1):
                select_set.append(i)
            else:
                self.beta[i] = 0
        
        for i in select_set:
            self.round_final_weight[i] = 0
            for j in range(0,math.ceil(self.select_frac * self.idx)):
                self.round_final_weight[i] += self.clients_weight[i][rd.randint(1,self.idx)]
        clients_sum = 0
        for i in select_set:
            clients_sum += self.round_final_weight[i]
        
        for i in select_set:
            self.round_final_weight[i] = self.round_final_weight[i]/clients_sum*(len(select_set)/(len(select_set)+1))
            # print(i)
            # print(f":{self.round_final_weight[i]}\n")
        self.round_final_weight[0] = 1.0/(len(select_set)+1)
        select_set.append(0)
        for i in select_set:
            for key in self.model.state_dict():
                self.model.state_dict()[key] += self.round_final_weight[i] * self.para_cache[i][key]

    def pull(self, client_nums, total):
        # for i in range(self.n_clients):
        #     for key in self.para_cache[0]:
        #         self.para_cache[i][key] -= self.model.state_dict()[key]
        # clear_parameter(self.model)
        self.idx += 1
        print(f"merge epoch : {self.idx}")
        self.mys()