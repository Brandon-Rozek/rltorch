from random import randrange
import torch
class ArgMaxSelector:
    def __init__(self, model, action_size, device = None):
        self.model = model
        self.action_size = action_size
        self.device = device
    def random_act(self):
        return randrange(self.action_size)
    def best_act(self, state):
        with torch.no_grad():
            if self.device is not None:
                self.device.to(self.device)
            action_values = self.model(state).squeeze(0)
            action = self.random_act() if (action_values[0] == action_values).all() else action_values.argmax().item()
        return action
    def act(self, state):
        return self.best_act(state)