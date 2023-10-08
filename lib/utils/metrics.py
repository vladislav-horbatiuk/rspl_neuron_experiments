import torch


def mse(prediction, target):
    return torch.square(prediction - target).mean()


def mae(prediction, target):
    return torch.abs(prediction - target).mean()


def mape(prediction, target, zero_eps=0.01):
    target = target.clone()
    target[torch.abs(target) < zero_eps] = zero_eps
    return torch.abs((prediction - target) / target).mean()


def print_metrics(prediction, target, name):
    print(f'Metrics for {name}:')
    print(f'\t MSE: {mse(prediction, target)}')
    print(f'\t MAE: {mae(prediction, target)}')
    print(f'\t MAPE: {mape(prediction, target)}')
