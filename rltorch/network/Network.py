class Network:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model, optimizer, config, logger = None, name = ""):
        self.model = model
        self.optimizer = optimizer(model.parameters(), lr = config['learning_rate'], weight_decay = config['weight_decay'])
        self.logger = logger
        self.name = name

    def __call__(self, *args):
        return self.model(*args)
        
    def clamp_gradients(self):
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
    
    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        self.optimizer.step()
    
    def log_named_parameters(self):
        if self.logger is not None:
            for name, param in self.model.named_parameters():
                self.logger.append(self.name + "/" + name, param.cpu().detach().numpy())

    
