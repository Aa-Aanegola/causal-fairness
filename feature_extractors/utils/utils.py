from torch.optim import Adam
from torch.nn import MSELoss, L1Loss, BCELoss

def model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 ** 2)

def get_optimizer(model, config):
    if config["training"]["optimizer"] == "adam":
        return Adam(model.parameters(), lr=config["training"]["lr"])
    else:
        raise ValueError("Invalid optimizer")

def get_loss(config):
    if config["training"]["loss"] == "mse":
        return MSELoss()
    elif config["training"]["loss"] == "l1":
        return L1Loss()
    elif config["training"]["loss"] == "bce":
        return BCELoss()
    else:
        raise ValueError("Invalid loss function")