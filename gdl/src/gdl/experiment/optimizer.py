import torch

def get_optimizer(name, model, lr, weight_decay=0):
    if isinstance(name, str):
        name = name.lower()
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
    
    parameters = [
        {'params': model.non_reg_params, 'weight_decay': 0},
        {'params': model.reg_params, 'weight_decay': weight_decay}
    ]
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
