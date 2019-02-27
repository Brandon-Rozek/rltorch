from random import randrange
import torch
from torch.distributions import Categorical
import rltorch
from rltorch.action_selector import ArgMaxSelector

class StochasticSelector(ArgMaxSelector):
    def __init__(self, model, action_size, memory = None, device = None):
        super(StochasticSelector, self).__init__(model, action_size, device = device)
        self.model = model
        self.action_size = action_size
        self.device = device
        self.memory = memory
    def best_act(self, state, log_prob = True):
        if self.device is not None:
            state = state.to(self.device)
        action_probabilities = self.model(state)
        distribution = Categorical(action_probabilities)
        action = distribution.sample()
        if log_prob and isinstance(self.memory, rltorch.memory.EpisodeMemory):
            self.memory.append_log_probs(distribution.log_prob(action))
        return action.item()