import torch


def clear_parameter(model: torch.nn.Module):
    """
    清空模型参数
    :param model:
    :return:
    """
    for key in model.state_dict():
        torch.nn.init.zeros_(model.state_dict()[key])
