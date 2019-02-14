class Network:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model, optimizer, config, device = None, logger = None, name = ""):
        self.model = model
        if 'weight_decay' in config:
            self.optimizer = optimizer(model.parameters(), lr = config['learning_rate'], weight_decay = config['weight_decay'])
        else:
            self.optimizer = optimizer(model.parameters(), lr = config['learning_rate'])
        self.logger = logger
        self.name = name
        self.device = device
        if self.device is not None:
            self.model = self.model.to(device)

    def __call__(self, *args):
        return self.model(*args)

    def clamp_gradients(self, x = 1):
        assert x > 0
        for param in self.model.parameters():
            param.grad.data.clamp_(-x, x)
    
    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        self.optimizer.step()
    
    def log_named_parameters(self):
        if self.logger is not None:
            for name, param in self.model.named_parameters():
                self.logger.append(self.name + "/" + name, param.cpu().detach().numpy())

    
