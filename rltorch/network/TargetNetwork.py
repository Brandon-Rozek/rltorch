from copy import deepcopy

class TargetNetwork:
    """
    Creates a clone of a network with syncing capabilities.

    Parameters
    ----------
    network
      The network to clone.
    device
      The device to put the cloned parameters in.
    """
    def __init__(self, network, device = None):
        self.model = network.model
        self.target_model = deepcopy(network.model)
        if device is not None:
            self.target_model = self.target_model.to(device)
        elif network.device is not None:
            self.target_model = self.target_model.to(network.device)

    def __call__(self, *args):
        return self.model(*args)

    def sync(self):
        """
        Perform a full state sync with the originating model.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def partial_sync(self, tau):
        """
        Partially move closer to the parameters of the originating
        model by updating parameters to be a mix of the
        originating and the clone models.
        
        Parameters
        ----------
        tau : number
          A number between 0-1 which indicates the proportion of the originator and clone in the new clone.
        """
        assert isinstance(tau, float)
        assert 0.0 < tau <= 1.0
        model_state = self.model.state_dict()
        target_state = self.target_model.state_dict()
        for grad_index, grad in model_state.items():
            target_state[grad_index].copy_((1 - tau) * target_state[grad_index] + tau * grad)
        self.target_model.load_state_dict(target_state)