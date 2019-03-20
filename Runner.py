import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import logging
import os
import time

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
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def learning_rate_decay(lr, batch_size, batch_denom, batches_per_opoch, boundary_epochs, decay_rates):
    initial_learning_rate = lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]

    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
    
    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

if __name__ == '__main__':
    hyperparams = get_hyperparams()
    device = 'cpu' # running on macbook for now

    # 1. setup logger
    logger = setup_logger()
    
    # 2. fetch data
    data_loader = fetch_data()
    data_gen = inf_generator(data_loader)
    batches_per_epoch = len(data_loader)

    # 3. setup network
    downsampling_layers = downsample_layers()
    feature_layers = [ODEBlock(ODEFunc(64), hyperparams['tol'])]
    fc_layers = fc_layers()
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)
    logger.info(model)
    
    # 4. run training
    learning_rate = learning_rate_decay(
        lr=hyperparams['lr'],
        batch_size=hyperparams['batch_size'],
        batch_denom=128,
        batches_per_opoch=batches_per_epoch,
        boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001])
    optimizer = torch.optim.SGD(
            model.parameters(),
            hyperparams['lr'])
    
    criterion = nn.CrossEntropyLoss().to(device)
    batch_time_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(hyperparams['nepochs'] * batches_per_epoch):
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate(itr)
        
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        nfe_forward = feature_layers[0].nfe
        feature_layers[0].nfe = 0
        
        loss.backward()
        optimizer.step()

        batch_time_meter.update(tim.time() - end)
           
    # 5. save model
