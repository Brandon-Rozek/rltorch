from .ArgMaxSelector import ArgMaxSelector
import torch
class IdentitySelector(ArgMaxSelector):
    def __init__(self, model, action_size, device = None):
        super(IdentitySelector, self).__init__(model, action_size, device = device)
    # random_act is already implemented in ArgMaxSelector
    def best_act(self, state):
        with torch.no_grad():
            if self.device is not None:
                state = state.to(self.device)
            action = self.model(state).squeeze(0).item()
        return action
    def act(self, state):
        return self.best_act(state)