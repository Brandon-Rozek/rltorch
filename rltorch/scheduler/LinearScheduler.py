from .Scheduler import Scheduler
class LinearScheduler(Scheduler):
    r"""
    A linear scheduler that given a certain number
    of iterations, equally spaces the values between
    a start and an end point.

    Notes
    -----
    The forumula used to produce the value :math:`y` is based on the number of
    times you call `next`. (denoted as :math:`i`)

    :math:`y(1) = initial\_value`

    :math:`y(i) = slope(i - 1) + y(1)`
    where :math:`slope = \frac{end\_value - initial\_value}{iterations}`.

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
        super(LinearScheduler, self).__init__(initial_value, end_value, iterations)
        self.slope = (end_value - initial_value) / iterations
    def __next__(self):
        if self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            return self.slope * (self.current_iteration - 1) + self.initial_value
        else:
            return self.end_value
        