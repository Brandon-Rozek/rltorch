import random
from collections import namedtuple
import torch
Transition = namedtuple('Transition',
    ('state', 'action', 'reward', 'next_state', 'done'))

class EpisodeMemory(object):
    """
    Memory structure that stores an entire episode and
    the observation's associated log-based probabilities.
    """
    def __init__(self):
        self.memory = []
        self.log_probs = []

    def append(self, *args):
        """
        Adds a transition to the memory.

        Parameters
        ----------
          *args
             The state, action, reward, next_state, done tuple
        """
        self.memory.append(Transition(*args))
    
    def append_log_probs(self, logprob):
        """
        Adds a log-based probability to the observation.
        """
        self.log_probs.append(logprob)

    def clear(self):
        """
        Clears the transitions and log-based probabilities.
        """
        self.memory.clear()
        self.log_probs.clear()

    def recall(self):
        """
        Return a list of the transitions with their 
        associated log-based probabilities.
        """
        if len(self.memory) != len(self.log_probs):
            raise ValueError("Memory and recorded log probabilities must be the same length.")
        return list(zip(*tuple(zip(*self.memory)), self.log_probs))

    def __len__(self):
        return len(self.memory)

    def __iter__(self):
        return iter(self.memory)

    def __contains__(self, value):
        return value in self.memory

    def __getitem__(self, index):
        return self.memory[index]

    def __setitem__(self, index, value):
        self.memory[index] = value

    def __reversed__(self):
        return reversed(self.memory)
