import yaml

from ..clients import *
from ...core.trainers import *


class Config:
    def __init__(self, path):
        self.path = path
        self.config = self.read_config()

    def read_config(self):
        with open(self.path, 'r') as f:
            file_data = f.read()
        return yaml.load(file_data, Loader=yaml.FullLoader)

    def run_mp(self):
        n_clients = self.config["n_clients"]
        cluster_conf = {
            "ps": (self.config["server"]["ip"], self.config["server"]["port"]),
            "client": [
                (self.config["client" + str(i)]["ip"], self.config["client" + str(i)]["port"])
                for i in range(1, n_clients + 1)
            ]
        }
        trainer = TrainerMP(
            cluster_conf=cluster_conf,
            optimizer=self.config["optimizer"]["type"],
            model=self.config["model"],
            data=self.config["data"],
            lr=self.config["optimizer"]["lr"],
            batch_size=self.config["batch_size"],
            path=self.config["path"],
            alpha=self.config["alpha"],
            local_epoch=self.config["local_epoch"],
            global_epoch=self.config["global_epoch"],
            device=self.config["device"],
            algorithm=self.config["algorithm"],
        )
        trainer.run()

    def run_distributed(self):
        if self.config["role"] == "server":
            data = all_data[self.config["data"]](
                n_clients=self.config["n_clients"],
                batch_size=self.config["batch_size"],
                path=self.config["path"],
                alpha=self.config["alpha"]
            )
            for i in range(self.config["n_clients"]):
                torch.save(data.trainLoader[i].dataset, f"./distributed_data/{self.config['data']}_{i + 1}")
            parameter_server = all_server[self.config["algorithm"]](
                ip=self.config["ip"],
                port=self.config["port"],
                global_epoch=self.config["global_epoch"],
                n_clients=self.config["n_clients"],
                model=self.config["model"],
                data=data.validationLoader,
                n_classes=len(data.train_set.classes),
                device=self.config["device"],
            )
            parameter_server.run()
        else:
            from torch.utils.data import DataLoader
            data = torch.load(f"./distributed_data/{self.config['data']}_{self.config['idx']}")
            loader = DataLoader(
                dataset=data,
                batch_size=self.config["batch_size"],
                shuffle=True,
            )
            cli = all_client[self.config["algorithm"]](
                ip=self.config["ip"],
                port=self.config["port"],
                server_ip=self.config["server_ip"],
                server_port=self.config["server_port"],
                model=self.config["model"],
                data=loader,
                sample_num=len(data),
                n_classes=len(data.dataset.classes),
                global_epoch=self.config["global_epoch"],
                local_epoch=self.config["local_epoch"],
                optimizer=self.config["optimizer"],
                lr=self.config["lr"],
                device=self.config["device"],
            )
            cli.run()
