import torch


class SimpleCNN(torch.nn.Module):
    """
    用于测试，只能用于MNIST数据集
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 3))  # 4 * 13 * 13
        self.conv2 = torch.nn.Conv2d(4, 16, (4, 4))  # 16 * 5 * 5
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))  # 32 * 3 * 3
        self.fc = torch.nn.Linear(32 * 9, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))
