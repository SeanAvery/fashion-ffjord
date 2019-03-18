import torch
import logging
import os

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

if __name__ == '__main__':
    # 1. setup logger
    setup_logger()
    # 2. setup data directories
    # 3. setup network
    # 4. run training
    # 5. save model
