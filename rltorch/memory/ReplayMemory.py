from random import sample
from collections import namedtuple
import torch
Transition = namedtuple('Transition',
    ('state', 'action', 'reward', 'next_state', 'done'))

# Implements a Ring Buffer
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def append(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def clear(self):
        self.memory.clear()
        self.position = 0

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

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

def zip_batch(minibatch):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)
        
    state_batch = torch.cat(state_batch)
    action_batch = torch.tensor(action_batch)
    reward_batch = torch.tensor(reward_batch)
    not_done_batch = ~torch.tensor(done_batch)
    next_state_batch = torch.cat(next_state_batch)[not_done_batch]

    return state_batch, action_batch, reward_batch, next_state_batch, not_done_batch