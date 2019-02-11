from .Scheduler import Scheduler
class LinearScheduler(Scheduler):
    def __init__(self, initial_value, end_value, iterations):
        super(LinearScheduler, self).__init__(initial_value, end_value, iterations)
        self.slope = (end_value - initial_value) / iterations
    def __next__(self):
        if self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            return self.slope * (self.current_iteration - 1) + self.initial_value
        else:
            return self.end_value
        