import numpy as np

def get_hyperparams():
    hyperparams = {
        batch_size: 1000,
        test_batch_size: 1000,
        tol: 1e-3, # type float
        save: True,
        display: True
    }
