from copy import deepcopy
import numpy as np
import torch
import torch.multiprocessing as mp
from .Network import Network
import rltorch.log as log

class fn_copy:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
    def __call__(self, x):
        return self.fn(x, *(self.args))

# [TODO] Should we torch.no_grad the __call__?
# What if we want to sometimes do gradient descent as well?
class ESNetworkMP(Network):
    """
    Network that functions from the paper Evolutionary Strategies (https://arxiv.org/abs/1703.03864)
    fitness_fun := model, *args -> fitness_value (float)
    We wish to find a model that maximizes the fitness function
    """
    def __init__(self, model, optimizer, population_size, fitness_fn, config, sigma=0.05, device=None, name=""):
        super(ESNetworkMP, self).__init__(model, optimizer, config, device, name)
        self.population_size = population_size
        self.fitness = fitness_fn
        self.sigma = sigma
        assert self.sigma > 0
        mp_ctx = mp.get_context("spawn")
        #[TODO] Probably should make number of processes a config variable
        self.pool = mp_ctx.Pool(processes=2) 

    # We're not going to be calculating gradients in the traditional way
    # So there's no need to waste computation time keeping track
    def __call__(self, *args):
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
        ## Generate Noise
        white_noise_dict, noise_dict = self._generate_noise_dicts()
        
        ## Generate candidate solutions
        candidate_solutions = self._generate_candidate_solutions(noise_dict)
        
        ## Calculate fitness then mean shift, scale
        fitness_values = torch.tensor(
            list(self.pool.map(fn_copy(self.fitness, args), candidate_solutions)),
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
        
        del white_noise_dict, noise_dict, candidate_solutions
    
    # To address error that you can't pickle pool objects...
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
