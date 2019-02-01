from copy import deepcopy
# Derived from ptan library
class TargetNetwork:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, network):
        self.model = network.model
        self.target_model = deepcopy(network.model)

    def __call__(self, *args):
        return self.model(*args)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def partial_sync(self, tau):
        """
        Blend params of target net with params from the model
        :param tau:
        """
        assert isinstance(tau, float)
        assert 0.0 < tau <= 1.0
        model_state = self.model.state_dict()
        target_state = self.target_model.state_dict()
        for grad_index, grad in model_state.items():
            target_state[grad_index].copy_((1 - tau) * target_state[grad_index] + tau * grad)
        self.target_model.load_state_dict(target_state)