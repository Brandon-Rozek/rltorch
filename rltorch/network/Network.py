class Network:
    """
    Wrapper around model and optimizer in PyTorch to abstract away common use cases.
    
    Parameters
    ----------
    model : nn.Module
      A PyTorch nn.Module.
    optimizer
      A PyTorch opimtizer from torch.optim.
    config : dict
      A dictionary of configuration items.
    device
      A device to send the weights to.
    logger
      Keeps track of historical weights
    name
      For use in logger to differentiate in analysis.
    """
    def __init__(self, model, optimizer, config, device=None, logger=None, name=""):
        self.model = model
        if 'weight_decay' in config:
            self.optimizer = optimizer(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        else:
            self.optimizer = optimizer(model.parameters(), lr=config['learning_rate'])
        self.logger = logger
        self.name = name
        self.device = device
        if self.device is not None:
            self.model = self.model.to(device)

    def __call__(self, *args):
        return self.model(*args)

    def clamp_gradients(self, x=1):
        """
        Forcing gradients to stay within a certain interval
        by setting it to the bound if it goes over it.

        Parameters
        ----------
        x : number > 0
          Sets the interval to be [-x, x]
        """
        assert x > 0
        for param in self.model.parameters():
            param.grad.data.clamp_(-x, x)
    
    def zero_grad(self):
        """
        Clears out gradients held in the model.
        """
        self.model.zero_grad()

    def step(self):
        """
        Run a step of the optimizer on `model`.
        """
        self.optimizer.step()
    
    def log_named_parameters(self):
        if self.logger is not None:
            for name, param in self.model.named_parameters():
                self.logger.append(self.name + "/" + name, param.cpu().detach().numpy())

    
