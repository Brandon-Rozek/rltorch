from collections import Counter
import numpy as np
import torch

class Logger:
    def __init__(self):
        self.log = {}
    def append(self, tag, value):
        if tag not in self.log.keys():
            self.log[tag] = []
        self.log[tag].append(value)
    def clear(self):
        self.log.clear()
    def keys(self):
        return self.log.keys()
    def __len__(self):
        return len(self.log)
    def __iter__(self):
        return iter(self.log)
    def __contains__(self, value):
        return value in self.log
    def __getitem__(self, index):
        return self.log[index]
    def __setitem__(self, index, value):
        self.log[index] = value
    def __reversed__(self):
        return reversed(self.log)

# Workaround since we can't use SummaryWriter in a different process
# class LogWriter:
#     def __init__(self, logger, writer):
#         self.logger = logger
#         self.writer = writer
#         self.steps = Counter()
#     def write(self):
#         for key in self.logger.keys():
#             for value in self.logger[key]:
#                 self.steps[key] += 1
#                 if isinstance(value, int) or isinstance(value, float):
#                     self.writer.add_scalar(key, value, self.steps[key])
#                 if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
#                     self.writer.add_histogram(key, value, self.steps[key])
#         self.logger.log = {}
#     def close(self):
#         self.writer.close()
    

class LogWriter:
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
    
