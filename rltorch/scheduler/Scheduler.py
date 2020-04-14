class Scheduler():
    def __init__(self, initial_value, end_value, iterations):
        self.initial_value = initial_value
        self.end_value = end_value
        self.max_iterations = iterations
        self.current_iteration = 0
    def __iter__(self):
        return self
    def __next__(self):
        raise NotImplementedError("__next__ not implemented in Scheduler")
