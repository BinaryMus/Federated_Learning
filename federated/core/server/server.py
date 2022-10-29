import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader


class BaseServer:
    """
    FedAvg push和pull 都是传递参数
    """

    def __init__(self,
                 epoch: int,
                 clients: List,
                 model: torch.nn.Module,
                 data: DataLoader,
                 device: str):
        self.clients = clients  # 所有的客户端
        self.epoch = epoch  # 全局epoch
        self.n_clients = len(clients)  # 客户端个数
        self.data = data  # 测试集
        self.device = device
        self.model = model.to(self.device)  # 全局模型

    def pull(self, client_nums, total):
        """
        接受clients参数并聚合
        :return:
        """
        clear_parameter(self.model)
        for key in self.model.state_dict():
            dtype = self.clients[0].model.state_dict()[key].dtype
            for idx in range(self.n_clients):
                self.model.state_dict()[key] += (
                        (client_nums[idx] / total) * self.clients[idx].model.state_dict()[key]).to(dtype)

    def push(self):
        """
        发布全局模型
        :return:
        """
        for idx in range(self.n_clients):
            clear_parameter(self.clients[idx].model)
            for key in self.clients[idx].model.state_dict():
                self.clients[idx].model.state_dict()[key] += self.model.state_dict()[key]

    def pull_push(self, client_nums, total):
        self.pull(client_nums, total)
        self.push()
        return self.validate()

    def validate(self):
        """
        全局验证
        :return:
        """
        total = 0
        correct1 = 0
        correct2 = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.data):
                total += len(x)
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                predict = output.argmax(dim=1)
                correct1 += torch.eq(predict, y).sum().float().item()
                y_resize = y.view(-1, 1)
                _, predict = output.topk(5)
                correct2 += torch.eq(predict, y_resize).sum().float().item()
        return correct1 / total, correct2 / total
