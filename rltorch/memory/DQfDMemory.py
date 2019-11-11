from .PrioritizedReplayMemory import PrioritizedReplayMemory
from collections import namedtuple

Transition = namedtuple('Transition',
    ('state', 'action', 'reward', 'next_state', 'done'))


class DQfDMemory(PrioritizedReplayMemory):
    def __init__(self, capacity, alpha):
        super().__init__(capacity, alpha)
        self.demo_position = 0
        self.obtained_transitions_length = 0
    
    def append(self, *args, **kwargs):
        super().append(*args, **kwargs)
        # Don't overwrite demonstration data
        self.position = self.demo_position + ((self.position + 1) % (len(self.memory) - self.demo_position))
    
    def append_demonstration(self, *args):
        demonstrations = self.memory[:self.demo_position]
        obtained_transitions = self.memory[self.demo_position:]
        if len(demonstrations) + 1 > self.capacity:
            self.memory.pop(0)
            self.memory.append(Transition(*args))
        else:
            if len(demonstrations) + len(obtained_transitions) + 1 > self.capacity:
                obtained_transitions = obtained_transitions[:(self.capacity - len(demonstrations) - 1)]
            self.memory = demonstrations + [Transition(*args)] + obtained_transitions
            self.demo_position += 1
