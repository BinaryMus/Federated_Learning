import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
from . import server


class Median(server.BaseServer):
    def __init__(self, epoch: int, clients: List, model: torch.nn.Module, data: DataLoader, device: str):
        super().__init__(epoch, clients, model, data, device)
        self.para_cache = []
        for i in range(self.n_clients):
            self.para_cache.append(clients[i].model.state_dict())
        self.dim =  0
        for key in self.model.state_dict():
            self.dim += self.model.state_dict()[key].view(-1).size(0)
        print(self.dim)

    def get_score_min(self):
        sum_score = 0.0
        for i in range(self.n_clients):
            for j in range(i+1,self.n_clients):
                # tensor_1 = torch.zeros(self.dim)
                # tensor_2 = torch.zeros(self.dim)
                # tensor_1 = []
                # tensor_2 = []
                tmp = 0
                for key in self.para_cache[i]:
                    for m, n, p in zip(self.para_cache[i][key], self.para_cache[j][key], self.model.state_dict()[key]):
                        for x, y, z in zip(m.view(-1), n.view(-1), p.view(-1)):
                            tmp += (x.item() - y.item()) ** 2
                tmp **= 0.5
                self.tmp_scores[i].append(tmp)
                self.tmp_scores[j].append(tmp)

                        # tensor_1.append(k)
                    # for k in self.para_cache[j][key]:
                        # tensor_2.append(k)
                # tensor_1 = self.para_cache[i][key].reshape(-1)
                # tensor_2 = self.para_cache[j][key].reshape(-1)
                # for k in range(dim):
                #     tmp_score = (tensor_1[k] - tensor_2[k])*(tensor_1[k] - tensor_2[k])
                #     self.scores[i][1] += tmp_score
                #     self.scores[j][1] += tmp_score
        for i in range(self.n_clients):
            sorted(self.tmp_scores[i])
            self.scores[i][1] = sum(self.tmp_scores[i][:self.n_clients-self.f-2])
        sorted(self.scores,key = lambda x:x[1])

    def pull(self, client_nums, total):
        clear_parameter(self.model)
        self.get_score_min()
        # for key in self.model.state_dict():
        #     dtype = self.clients[0].model.state_dict()[key].dtype
        #     for idx in range(self.n_clients):
        #         self.model.state_dict()[key] += (
        #                 (client_nums[idx] / total) * self.clients[idx].model.state_dict()[key]).to(dtype)
        for key in self.model.state_dict():
            dtype = self.clients[0].model.state_dict()[key].dtype
            self.model.state_dict()[key] += (self.clients[self.scores[0][0]].model.state_dict()[key]).to(dtype)
        # self.model.load_state_dict(self.para_cache[self.scores[0][0]])
        # acc1, acc5 = self.validate()
        '''
        print(f"SERVER@{self}:{self.port} INFO: "
              f"Global Epoch[{self.round}|{self.global_epoch}]"
              f"Top-1 Accuracy: {acc1} "
              f"Top-5 Accuracy: {acc5}")
        '''