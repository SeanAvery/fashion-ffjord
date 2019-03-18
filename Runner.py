import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import logging
import os

from ODE import ODEFunc, ODEBlock, conv, norm

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

def fetch_data():
    # setup tensor transformer
    data_transform = transforms.Compose([ transforms.ToTensor() ])   

    data_loader = DataLoader(
            datasets.MNIST(root='./data/mnist', train=True, download=True, transform=data_transform),
            batch_size=1000,
            shuffle=True,
            num_workers=4,
            drop_last=True)

if __name__ == '__main__':
    # 1. setup logger
    setup_logger()
    # 2. fetch data
    fetch_data()
    # 3. setup network
    network = [ODEBlock(ODEFunc(64))]
    # 4. run training
    # 5. save model
