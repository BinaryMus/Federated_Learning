from ast import NotEq
from turtle import distance
from . import server

class Multi_Krum(server.BaseServer):
    def __init__(self, ip: str, port: int, global_epoch: int, n_clients: int, model: str, data: DataLoader, n_classes: int, device: str, scores: list):
        super().__init__(ip, port, global_epoch, n_clients, model, data, n_classes, device)
        scores = [[0,0.0]]
        agg_numbers = (n_clients+2) // 3
        for i in range(n_clients):
            scores[i][0] = i

    def get_score_min(self):
        sum_score = 0.0
        for key in self.model.state_dict():
            for i in range(self.n_clients):
                for j in range(i+1,self.n_clients):
                    tensor_1 = self.para_cache[i][1].reshape(-1)
                    tensor_2 = self.para_cache[j][1].reshape(-1)
                    dim = tensor_1.shape
                    for k in range(dim):
                        tmp_score = (tensor_1[k] - tensor_2[k])*(tensor_1[k] - tensor_2[k])
                        self.scores[i][1] += tmp_score
                        self.scores[j][1] += tmp_score
        sorted(scores,key = lambda x:x[1])

    def aggregate(self):
        clear_parameter(self.model)
        get_score_min()
        for key in self.model.state_dict():
            dtype = self.para_cache[0][1][key].dtype
            for idx in range(agg_numbers):
                self.model.state_dict()[key] += \
                    (1.0 / agg_numbers) * self.para_cache[scores[idx][0]][1][key].to(dtype)
        acc1, acc5 = self.validate()
        ''' print(f"SERVER@{self.ip}:{self.port} INFO: "
              f"Global Epoch[{self.round}|{self.global_epoch}]"
              f"Top-1 Accuracy: {acc1} "
              f"Top-5 Accuracy: {acc5}")
        '''