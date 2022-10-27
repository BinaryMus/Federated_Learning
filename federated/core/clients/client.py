import torch
from torch.utils.data import DataLoader


class Client:
    def __init__(self,
                 index: int,
                 n_clients: int,
                 model: torch.nn.Module,
                 data: DataLoader,
                 epoch: int,
                 optimizer: str,
                 lr: float,
                 device: str,
                 criterion=torch.nn.CrossEntropyLoss()):
        self.index = index  # 客户端索引
        self.n_clients = n_clients  # 总客户端数量
        self.criterion = criterion  # 损失函数
        self.data = data  # 数据
        self.device = torch.device(device)  # 设备
        self.model = model.to(device)  # 模型
        self.lr = lr  # 学习率
        self.epoch = epoch  # 本地多轮迭代次数
        self.loss = []  # 本地训练的损失
        if optimizer == "SGD":  # 优化器
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            assert True, "暂无其他优化器"

    def train(self):
        """
        一轮的训练
        :return: 返回该轮的平均loss
        """
        loss_avg = 0
        cnt = 0
        for idx, (x, y) in enumerate(self.data):
            cnt += 1
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            loss_avg += loss.item()
            self.optimizer.step()
        return loss_avg / cnt

    def train_loop(self):
        """
        本地多轮迭代
        :return:
        """
        for epoch in range(self.epoch):
            loss_avg = self.train()
            self.loss.append(loss_avg)
            print(f"{str(self)} Local Epoch[{epoch}|{self.epoch}] Loss:{round(loss_avg, 3)}")

    def __str__(self):
        return f"Client[{self.index + 1}|{self.n_clients}]"
