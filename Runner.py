import torch
import logging

def setup_logger(logpath, filepath, displaying=True, saving=True, debug=False):
    # instantiate logger object
    logger = logging.getLogger()
    
    # setup logging level
    if debug: 
        level = logging.DEBUG
    else: 
        level = logging.INFO
    logger.setLevel(level)

if __name__ == '__main__':
    # 1. setup logger
    setup_logger('./', './')
    # 2. setup data directories
    # 3. setup network
    # 4. run training
    # 5. save model
