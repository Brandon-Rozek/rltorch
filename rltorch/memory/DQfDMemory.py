from .PrioritizedReplayMemory import PrioritizedReplayMemory
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
    ('state', 'action', 'reward', 'next_state', 'done'))


class DQfDMemory(PrioritizedReplayMemory):
    def __init__(self, capacity, alpha, max_demo = -1):
        assert max_demo <= capacity
        super().__init__(capacity, alpha)
        self.demo_position = 0
        self.max_demo = max_demo # -1 means no maximum number of demonstrations
    
    def append(self, *args, **kwargs):
        last_position = self.position # Get position before super classes change it
        super().append(*args, **kwargs)
        # Don't overwrite demonstration data
        new_position = ((last_position - self.demo_position + 1) % (self.capacity - self.demo_position))
        self.position = self.demo_position + new_position
    
    def append_demonstration(self, *args):
        demonstrations = self.memory[:self.demo_position]
        obtained_transitions = self.memory[self.demo_position:]
        max_demo = self.max_demo if self.max_demo > -1 else self.capacity
        if len(demonstrations) + 1 > max_demo:
            self.memory.pop(0)
            self.memory.append(Transition(*args))
            self._it_sum[len(self.memory) - 1] = self._max_priority ** self._alpha
            self._it_min[len(self.memory) - 1] = self._max_priority ** self._alpha
        else:
            if len(demonstrations) + len(obtained_transitions) + 1 > self.capacity:
                obtained_transitions = obtained_transitions[1:]
            self.memory = demonstrations + [Transition(*args)] + obtained_transitions
            self._it_sum[len(demonstrations)] = self._max_priority ** self._alpha
            self._it_min[len(demonstrations)] = self._max_priority ** self._alpha
            self.demo_position += 1
            self.position += 1
    
    def sample_n_steps(self, batch_size, steps, beta):
        assert beta > 0

        sample_size = batch_size // steps
        
        # Sample indexes and get n-steps after that
        idxes = self._sample_proportional(sample_size)
        step_idxes = []
        for i in idxes:
            # If the interval of experiences fall between demonstration and obtained, move it over to the demonstration half
            if i < self.demo_position and i + steps > self.demo_position:
                diff = i + steps - self.demo_position
                step_idxes += range(i - diff, i + steps - diff)
            elif i > steps:
                step_idxes += range(i - steps, i)
            else:
                step_idxes += range(i, i + steps)

        # Calculate appropriate weights and assign it to the values of the same sequence
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)
        for idx in step_idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        
        # Combine all the data together into a batch
        encoded_sample = tuple(zip(*self._encode_sample(step_idxes)))
        batch = list(zip(*encoded_sample, weights, step_idxes))
        return batch
