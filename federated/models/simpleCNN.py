from . import Model

import torch


class SimpleCNN(Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 3))  # 4 * 13 * 13
        self.conv2 = torch.nn.Conv2d(4, 16, (4, 4))  # 16 * 5 * 5
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))  # 32 * 3 * 3
        self.fc = torch.nn.Linear(32 * 9, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))
