import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset


class Data:
    def __init__(self, plot_path):
        self.train_set = None
        self.validate_set = None
        self.plot_path = plot_path

    def train_loader(self, alpha, n_clients, batch_size, flag=True):
        labels = np.array(self.train_set.targets)
        split_idx = self.split_non_iid(labels, alpha=alpha, n_clients=n_clients)
        if flag:
            plt.figure(figsize=(20, 3))
            plt.hist([labels[idc] for idc in split_idx], stacked=True,
                     bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
                     label=["Client {}".format(i + 1) for i in range(n_clients)], rwidth=0.5)
            plt.xticks(np.arange(len(self.train_set.classes)), self.train_set.classes)
            plt.legend()
            plt.savefig(self.plot_path + "/" + self.__class__.__name__ + "_data_distribution.png")

        client_nums = []
        data_loader = []

        for i in range(n_clients):
            client_nums.append(len(split_idx[i]))
            data_loader.append(DataLoader(
                dataset=Subset(self.train_set, split_idx[i]),
                batch_size=batch_size,
                shuffle=True
            ))
        return data_loader, client_nums, len(self.train_set.data)

    def validate_loader(self, batch_size):
        return DataLoader(self.validate_set, batch_size=batch_size, shuffle=False)

    @staticmethod
    def split_non_iid(train_labels, alpha, n_clients):
        n_classes = train_labels.max() + 1
        label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
        class_idx = [np.argwhere(train_labels == y).flatten()
                     for y in range(n_classes)]
        client_idx = [[] for _ in range(n_clients)]
        for c, f in zip(class_idx, label_distribution):
            for i, idx in enumerate(np.split(c, (np.cumsum(f)[:-1] * len(c)).astype(int))):
                client_idx[i] += [idx]
        client_idx = [np.concatenate(idx) for idx in client_idx]
        return client_idx
