from .ArgMaxSelector import ArgMaxSelector
import numpy as np 
class EpsilonGreedySelector(ArgMaxSelector):
    def __init__(self, model, action_size, device = None, epsilon = 0.1, epsilon_decay = 1, epsilon_min = 0.1):
        super(EpsilonGreedySelector, self).__init__(model, action_size, device = device)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
    # random_act is already implemented in ArgMaxSelector
    # best_act is already implemented in ArgMaxSelector
    def act(self, state):
        action = self.random_act() if np.random.rand() < self.epsilon else self.best_act(state)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        return action