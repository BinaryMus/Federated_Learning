from ..core.clients import *
from ..core.server import *
from ..datasets import *
from ..models import *

all_optim = {"SGD": torch.optim.SGD}
all_arch = {"SimpleCNN": SimpleCNN, "VGG11": VGG11, "ResNet18": Resnet18}
all_data = {"MNIST": Mnist, "CIFAR10": Cifar10}
all_server = {"FedAVG": BaseServer, "FedProx": BaseServer}
all_client = {"FedAVG": BaseClient, "FedProx": FedProxClient}
