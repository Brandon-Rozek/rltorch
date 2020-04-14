from copy import deepcopy
import numpy as np
import torch
from .Network import Network
import rltorch.log as log

# [TODO] Should we torch.no_grad the __call__?
# What if we want to sometimes do gradient descent as well?
class ESNetwork(Network):
    """
    Uses evolutionary tecniques to optimize a neural network.

    Notes
    -----
    Derived from the paper
    Evolutionary Strategies
    (https://arxiv.org/abs/1703.03864)

    Parameters
    ----------
    model : nn.Module
      A PyTorch nn.Module.
    optimizer
      A PyTorch opimtizer from torch.optim.
    population_size : int
      The number of networks to evaluate each iteration.
    fitness_fn : function
      Function that evaluates a network and returns a higher
      number for better performing networks.
    sigma : number
      The standard deviation of the guassian noise added to
      the parameters when creating the population.
    config : dict
      A dictionary of configuration items.
    device
      A device to send the weights to.
    name
      For use in logger to differentiate in analysis.
    """
    def __init__(self, model, optimizer, population_size, fitness_fn, config, sigma=0.05, device=None, name=""):
        super(ESNetwork, self).__init__(model, optimizer, config, device, name)
        self.population_size = population_size
        self.fitness = fitness_fn
        self.sigma = sigma
        assert self.sigma > 0

    def __call__(self, *args):
        """
        Notes
        -----
        Since gradients aren't going to be computed in the
        traditional fashion, there is no need to keep
        track of the computations performed on the
        tensors.
        """
        with torch.no_grad():
            result = self.model(*args)
        return result


    def _generate_noise_dicts(self):
        model_dict = self.model.state_dict()
        white_noise_dict = {}
        noise_dict = {}
        for key in model_dict.keys():
            white_noise_dict[key] = torch.randn(
                self.population_size,
                *model_dict[key].shape,
                device=self.device
            )
            noise_dict[key] = self.sigma * white_noise_dict[key]
        return white_noise_dict, noise_dict

    def _generate_candidate_solutions(self, noise_dict):
        model_dict = self.model.state_dict()
        candidate_solutions = []
        for i in range(self.population_size):
            candidate_statedict = {}
            for key in model_dict.keys():
                candidate_statedict[key] = model_dict[key] + noise_dict[key][i]
            candidate = deepcopy(self.model)
            candidate.load_state_dict(candidate_statedict)
            candidate_solutions.append(candidate)
        return candidate_solutions

    def calc_gradients(self, *args):
        """
        Calculate gradients by shifting parameters
        towards the networks with the highest fitness value.

        This is calculated by evaluating the fitness of multiple
        networks according to the fitness function specified in
        the class.
        """
        ## Generate Noise
        white_noise_dict, noise_dict = self._generate_noise_dicts()
        
        ## Generate candidate solutions
        candidate_solutions = self._generate_candidate_solutions(noise_dict)
        
        ## Calculate fitness then mean shift, scale
        fitness_values = torch.tensor(
            [self.fitness(x, *args) for x in candidate_solutions],
            device=self.device
        )
        if log.enabled:
            log.Logger[self.name + "/" + "fitness_value"].append(fitness_values.mean().item())
        fitness_values = (fitness_values - fitness_values.mean()) / (fitness_values.std() + np.finfo('float').eps)

        ## Insert adjustments into gradients slot
        self.zero_grad()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                noise_dim_n = len(white_noise_dict[name].shape)
                dim = np.repeat(1, noise_dim_n - 1).tolist() if noise_dim_n > 0 else []
                param.grad = (white_noise_dict[name] * fitness_values.float().reshape(self.population_size, *dim)).mean(0) / self.sigma
