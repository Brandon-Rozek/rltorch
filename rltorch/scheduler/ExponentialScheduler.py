from .Scheduler import Scheduler
class ExponentialScheduler(Scheduler):
    r"""
    A exponential scheduler that given a certain number
    of iterations, spaces the values between
    a start and an end point in an exponential order.

    Notes
    -----
    The forumula used to produce the value :math:`y` is based on the number of
    times you call `next`. (denoted as :math:`i`)

    :math:`y(1) = initial\_value`
    :math:`y(i) = y(1) \cdot base^{i - 1}`
    :math:`base = \sqrt[iterations]{\frac{end\_value}{initial\_value}}`.

    Another property is that :math:`y(iterations) = end\_value`.

    Parameters
    ----------
    initial_value : number
      The first value returned in the schedule.
    end_value: number
      The value returned when the maximum number of iterations are reached
    iterations: int
      The total number of iterations
    """
    def __init__(self, initial_value, end_value, iterations):
        super(ExponentialScheduler, self).__init__(initial_value, end_value, iterations)
        self.base = (end_value / initial_value) ** (1.0 / iterations)
    def __next__(self):
        if self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            return self.initial_value * (self.base ** (self.current_iteration - 1))
        else:
            return self.end_value

