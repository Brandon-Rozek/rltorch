from collections import Counter, defaultdict
from typing import Dict, List, Any
import numpy as np
import torch

Logger: Dict[Any, List[Any]] = defaultdict(list)

class LogWriter:
    """
    Takes a logger and writes it to a writter.
    While keeping track of the number of times it
    a certain tag.

    Notes
    -----
    Used to keep track of scalars and histograms in
    Tensorboard.

    Parameters
    ----------
    writer
      The tensorboard writer.
    """
    def __init__(self, writer):
        self.writer = writer
        self.steps = Counter()
    def write(self, logger):
        for key in logger.keys():
            for value in logger[key]:
                self.steps[key] += 1
                if isinstance(value, int) or isinstance(value, float):
                    self.writer.add_scalar(key, value, self.steps[key])
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    self.writer.add_histogram(key, value, self.steps[key])
        logger.clear()
    def close(self):
        self.writer.close()
    
