from os import environ
import numpy as np
import random
import torch

def set_seed(SEED):
    # Set `PYTHONHASHSEED` environment variable at a fixed value
    environ['PYTHONHASHSEED'] = str(SEED)

    np.random.seed(SEED)
    random.seed(SEED)

    # Pytorch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
