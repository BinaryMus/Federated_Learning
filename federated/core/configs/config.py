import yaml

from ...core.trainers import Trainer


class Config:
    def __init__(self, path):
        self.path = path
        self.config = self.read_config()

    def read_config(self):
        with open(self.path, 'r') as f:
            file_data = f.read()
        return yaml.load(file_data, Loader=yaml.FullLoader)

    def run(self):
        trainer = Trainer(n_clients=self.config["n_clients"],
                          optimizer=self.config["optimizer"]["type"],
                          model=self.config["model"],
                          data=self.config["data"],
                          lr=self.config["optimizer"]["lr"],
                          batch_size=self.config["batch_size"],
                          path=self.config["path"],
                          alpha=self.config["alpha"],
                          local_epoch=self.config["local_epoch"],
                          global_epoch=self.config["global_epoch"],
                          algorithm=self.config["algorithm"],
                          device=self.config["device"])
        trainer.train()
