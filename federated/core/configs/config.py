import yaml

from ...core.trainers import TrainerMP


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
        pass
