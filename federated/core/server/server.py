import torch
from typing import List
from ..utils import clear_parameter
from torch.utils.data import DataLoader
import struct,random
from scipy.stats import norm as spnorm


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
        self.z_max = -1.0
    
    def attack(self,attack_type,attack_ratio):
        attack_clients = min((int)(attack_ratio*self.n_clients)+1,self.n_clients)
        # print("attack:" + (str)(attack_clients))
        if attack_type == 0:
            return
        elif attack_type == 1:
            for i in range(1,attack_clients):
                self.reverse_direction_attack(self.clients[i].model.state_dict())
        elif attack_type == 2:
            for i in range(1,attack_clients):
                self.bit_flipping_attack(self.clients[i].model.state_dict())
        elif attack_type == 3:
            n = self.n_clients - 1
            m = attack_clients
            if self.z_max < 0:
                s = (n + 1) // 2 - m
                cdf_value = (n - m - s) / (n - m)
                self.z_max = spnorm.ppf(cdf_value)
                
            gradients = []
            for i in range(1, n - m + 1):
                gradients.append(torch.cat([value.view(-1) for value in self.clients[i].model.state_dict().values()]))
            stk_gradients = torch.stack(gradients, dim = 1)
            mu = torch.mean(stk_gradients, dim = 1)
            std = torch.std(stk_gradients, dim = 1)
            alie = mu - std * self.z_max
            # print(alie)
            
            model_state_dict = self.clients[n - m + 1].model.state_dict()
            
            start = 0
            for key, param in model_state_dict.items():
                end = start + param.numel()
                param.data = alie[start:end].view(param.shape)
                start = end
            # print(model_state_dict)
            for i in range(n - m + 1, self.n_clients):
                self.clients[i].model.load_state_dict(model_state_dict)

    # type = 1
    def reverse_direction_attack(self,model):
        for key in model:
            model[key] -= 2 * model[key]

    # type = 2

    def binary(self,num):
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

    def bin2float(self,bits):
        return struct.unpack('f',struct.pack('I',int(bits,2)))[0]

    def float_flipping(self,num,max_flipping_counts: int = 4):
        bin_str = list(self.binary(num))
        flipping_counts = random.randint(0,max_flipping_counts)
        # print(flipping_counts)
        for i in range(flipping_counts):
            rnd = random.randint(0,31)
            bin_str[rnd] = '1' if bin_str[rnd] == '0' else '0'
        return self.bin2float(''.join(bin_str))
    
    def bit_flipping_attack(self,model,max_flipping_counts: int = 4):
        for key in model:
            tensor_shape = model[key].shape
            tensor_size = model[key].view(-1).size(0)
            tensor_attacked = torch.zeros(tensor_size)
            idx = 0
            model_key = model[key].view(-1)
            for idx in range(tensor_size):
                tensor_attacked[idx] = self.float_flipping(model_key[idx],max_flipping_counts)
                idx += 1
            torch.nn.init.zeros_(model[key])
            model[key] += tensor_attacked.reshape(tensor_shape)
    
    # type = 3


    def pull(self, client_nums, total):
        """
        接受clients参数并聚合
        :return:
        """
        # clear_parameter(self.model)
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

    def pull_push(self, *args):
        self.pull(args[0], args[1])
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
