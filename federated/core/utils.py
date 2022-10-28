import os
import random

import numpy as np
import torch


def clear_parameter(model: torch.nn.Module):
    """
    清空模型参数
    :param model:
    :return:
    """
    for key in model.state_dict():
        torch.nn.init.zeros_(model.state_dict()[key])


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
