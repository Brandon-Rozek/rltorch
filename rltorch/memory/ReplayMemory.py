import random
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

    def _encode_sample(self, indexes):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indexes:
            observation = self.memory[i]
            state, action, reward, next_state, done = observation
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        batch = list(zip(states, actions, rewards, next_states, dones))
        return batch


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def sample_n_steps(self, batch_size, steps):
        idxes = random.sample(range(len(self.memory) - steps), batch_size // steps)
        step_idxes = []
        for i in idxes:
            step_idxes += range(i, i + steps)
        return self._encode_sample(step_idxes)

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

def zip_batch(minibatch, priority = False, want_indices = False):
    if priority:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indexes = zip(*minibatch)
    elif want_indices:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, indexes = zip(*minibatch)
    else:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)
        
    state_batch = torch.cat(state_batch)
    action_batch = torch.tensor(action_batch)
    reward_batch = torch.tensor(reward_batch)
    not_done_batch = ~torch.tensor(done_batch)
    next_state_batch = torch.cat(next_state_batch)

    if priority:
        return state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, weights, indexes
    elif want_indices:
        return state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, indexes
    else:
        return state_batch, action_batch, reward_batch, next_state_batch, not_done_batch