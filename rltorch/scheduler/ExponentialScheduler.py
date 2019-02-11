from .Scheduler import Scheduler
class ExponentialScheduler(Scheduler):
    def __init__(self, initial_value, end_value, iterations):
        super(ExponentialScheduler, self).__init__(initial_value, end_value, iterations)
        self.base = (end_value / initial_value) ** (1.0 / iterations)
    def __next__(self):
        if self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            return self.initial_value * (self.base ** (self.current_iteration - 1))
        else:
            return self.end_value

