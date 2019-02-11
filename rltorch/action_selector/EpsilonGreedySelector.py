from .ArgMaxSelector import ArgMaxSelector
import numpy as np 
class EpsilonGreedySelector(ArgMaxSelector):
    def __init__(self, model, action_size, device = None, epsilon = 0.1):
        super(EpsilonGreedySelector, self).__init__(model, action_size, device = device)
        self.epsilon = epsilon
    # random_act is already implemented in ArgMaxSelector
    # best_act is already implemented in ArgMaxSelector
    def act(self, state):
        eps = next(self.epsilon) if isinstance(self.epsilon, collections.Iterable) else self.epsilon
        action = self.random_act() if np.random.rand() < epsilon else self.best_act(state)
        return action