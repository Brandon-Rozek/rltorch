from random import randrange
class RandomSelector():
    def __init__(self, action_size):
        self.action_size = action_size
    def random_act(self):
        return randrange(action_size)
    def best_act(self, state):
        return self.random_act()
    def act(self, state):
        return self.random_act()
