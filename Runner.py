import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import logging
import os

from ODE import ODEFunc, ODEBlock, downsample_layers, fc_layers
from hyperparams import get_hyperparams

def setup_logger(displaying=True, saving=False, debug=False):
    # instantiate logger object
    logger = logging.getLogger()
    
    # setup logging level
    if debug: 
        level = logging.DEBUG
    else: 
        level = logging.INFO
    logger.setLevel(level)

    # setup console logging display
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

def fetch_data():
    # setup tensor transformer
    data_transform = transforms.Compose([ transforms.ToTensor() ])   

    data_loader = DataLoader(
            datasets.MNIST(root='./data/mnist', train=True, download=True, transform=data_transform),
            batch_size=1000,
            shuffle=True,
            num_workers=4,
            drop_last=True)
    return data_loader

def inf_generator(iterable):
    iterator.iterable.__iter__()
    while True:
        try: 
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def learning_rate_decay(lr, batch_size, batch_denom, batches_per_opoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

if __name__ == '__main__':
    hyperparams = get_hyperparams()

    # 1. setup logger
    logger = setup_logger()
    
    # 2. fetch data
    data_loader = fetch_data()
    data_gen = inf_generator(data_loader)
    
    # 3. setup network
    downsampling_layers = downsample_layers()
    feature_layers = [ODEBlock(ODEFunc(64), hyperparams['tol'])]
    fc_layers = fc_layers()
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)
    logger.info(model)
    
    # 4. run training
    data_gen = inf_generator(data_loader)
    batches_per_epoch = len(data_loader)
    print(batches_per_epoch)
    learning_rate = learning_rate_decay(
        lr=hyperparams['lr'],
        batch_size=hyperparams['batch_size'],
        batch_denom=128,
        batches_per_opoch=batches_per_epoch,
        boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001])

    # 5. save model
