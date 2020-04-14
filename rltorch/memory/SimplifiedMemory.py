from random import sample
from collections import deque

class ReplayMemory:
    """
    Creates a queue of a fixed size.

    Parameters
    ----------
    capacity : int
      The maximum size of the buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def append(self, **kwargs):
        """
        Adds a transition to the buffer.

        Parameters
        ----------
        **kwargs
          The state, action, reward, next_state, done tuple
        """
        self.memory.append(kwargs)

    def clear(self):
        """
        Clears the buffer.
        """
        self.memory.clear()

    def _encode_sample(self, indices):
        batch = list()
        for i in indices:
            batch.append(self.memory[i])
        return batch

    def sample(self, batch_size):
        """
        Returns a random sample from the buffer.

        Parameters
        ----------
        batch_size : int
          The number of observations to sample.
        """
        return sample(self.memory, batch_size)
    
    def sample_n_steps(self, batch_size, steps):
        r"""
        Returns a random sample of sequential batches of size steps.

        Notes
        -----
        The number of batches sampled is :math:`\lfloor\frac{batch\_size}{steps}\rfloor`.

        Parameters
        ----------
        batch_size : int
          The total number of observations to sample.
        steps : int
          The number of observations after the one selected to sample.
        """
        idxes = sample(
            range(len(self.memory) - steps),
            batch_size // steps
        )
        step_idxes = []
        for i in idxes:
            step_idxes += range(i, i + steps)
        return self._encode_sample(step_idxes)

    def __len__(self):
        return len(self.memory)
