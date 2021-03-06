from copy import deepcopy
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import rltorch
import rltorch.memory as M
import rltorch.log as log


# Q-Evolutionary Policy Agent
# Maximizes the policy with respect to the Q-Value function.
# Since function is non-differentiabile, depends on the Evolutionary Strategy algorithm
class QEPAgent:
    def __init__(self, policy_net, value_net, memory, config, target_value_net=None, entropy_importance=0, policy_skip=4):
        self.policy_net = policy_net
        assert isinstance(self.policy_net, rltorch.network.ESNetwork) or isinstance(self.policy_net, rltorch.network.ESNetworkMP)
        self.policy_net.fitness = self.fitness
        self.value_net = value_net
        self.target_value_net = target_value_net
        self.memory = memory
        self.config = deepcopy(config)
        self.policy_skip = policy_skip
        self.entropy_importance = entropy_importance
    
    def save(self, file_location):
        torch.save({
            'policy': self.policy_net.model.state_dict(),
            'value': self.value_net.model.state_dict()
            }, file_location)
    def load(self, file_location):
        checkpoint = torch.load(file_location)
        self.value_net.model.state_dict(checkpoint['value'])
        self.value_net.model.to(self.value_net.device)
        self.policy_net.model.state_dict(checkpoint['policy'])
        self.policy_net.model.to(self.policy_net.device)
        if self.target_value_net is not None:
            self.target_net.sync()

    def fitness(self, policy_net, value_net, state_batch):
        batch_size = len(state_batch)
        with torch.no_grad():
            action_probabilities = policy_net(state_batch)
        action_size = action_probabilities.shape[1]
        distributions = list(map(Categorical, action_probabilities))
        actions = torch.stack([d.sample() for d in distributions])
      
        with torch.no_grad():
            state_values = value_net(state_batch)

        # Weird hacky solution where in multiprocess, it sometimes spits out nans
        # So have it try again
        while torch.isnan(state_values).any():
            with torch.no_grad():
                state_values = value_net(state_batch)

        obtained_values = state_values.gather(1, actions.view(len(state_batch), 1)).squeeze(1)
        # return -obtained_values.mean().item()
        entropy_importance = 0 # Entropy accounting for 1% of loss seems to work well
        entropy_importance = next(self.entropy_importance) if isinstance(self.entropy_importance, collections.Iterable) else self.entropy_importance
        value_importance = 1 - entropy_importance
        
        # entropy_loss = (action_probabilities * torch.log2(action_probabilities)).sum(1) # Standard entropy loss from information theory
        entropy_loss = (action_probabilities - torch.tensor(1 / action_size, device=state_batch.device).repeat(len(state_batch), action_size)).abs().sum(1)
        
        return (entropy_importance * entropy_loss - value_importance * obtained_values).mean().item()
        

    def learn(self):
        if len(self.memory) < self.config['batch_size']:
            return
        
        if isinstance(self.memory, M.PrioritizedReplayMemory):
            weight_importance = self.config['prioritized_replay_weight_importance']
            # If it's a scheduler then get the next value by calling next, otherwise just use it's value
            beta = next(weight_importance) if isinstance(weight_importance, collections.Iterable) else weight_importance
            minibatch = self.memory.sample(self.config['batch_size'], beta=beta)
            state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, importance_weights, batch_indexes = M.zip_batch(minibatch, priority=True)
        else:
            minibatch = self.memory.sample(self.config['batch_size'])
            state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = M.zip_batch(minibatch)
        
        # Send to their appropriate devices
        state_batch = state_batch.to(self.value_net.device)
        action_batch = action_batch.to(self.value_net.device)
        reward_batch = reward_batch.to(self.value_net.device).float()
        next_state_batch = next_state_batch.to(self.value_net.device)
        not_done_batch = not_done_batch.to(self.value_net.device)

        state_values = self.value_net(state_batch)
        obtained_values = state_values.gather(1, action_batch.view(self.config['batch_size'], 1))

        with torch.no_grad():
            # Use the target net to produce action values for the next state
            # and the regular net to select the action
            # That way we decouple the value and action selecting processes (DOUBLE DQN)
            not_done_size = not_done_batch.sum()
            next_state_values = torch.zeros_like(state_values, device=self.value_net.device)
            if self.target_value_net is not None:
                next_state_values[not_done_batch] = self.target_value_net(next_state_batch[not_done_batch])
                next_best_action = self.value_net(next_state_batch[not_done_batch]).argmax(1)
            else:
                next_state_values[not_done_batch] = self.value_net(next_state_batch[not_done_batch])
                next_best_action = next_state_values[not_done_batch].argmax(1)

            best_next_state_value = torch.zeros(self.config['batch_size'], device=self.value_net.device)
            best_next_state_value[not_done_batch] = next_state_values[not_done_batch].gather(1, next_best_action.view((not_done_size, 1))).squeeze(1)
            
        expected_values = (reward_batch + (self.config['discount_rate'] * best_next_state_value)).unsqueeze(1)

        if isinstance(self.memory, M.PrioritizedReplayMemory):
            value_loss = (torch.as_tensor(importance_weights, device=self.value_net.device) * ((obtained_values - expected_values)**2).squeeze(1)).mean()
        else:
            value_loss = F.mse_loss(obtained_values, expected_values)
        
        if log.enabled:
            log.Logger["Loss/Value"].append(value_loss.item())

        self.value_net.zero_grad()
        value_loss.backward()
        self.value_net.clamp_gradients()
        self.value_net.step()

        if self.target_value_net is not None:
            if 'target_sync_tau' in self.config:
                self.target_value_net.partial_sync(self.config['target_sync_tau'])
            else:
                self.target_value_net.sync()

        if isinstance(self.memory, M.PrioritizedReplayMemory):
            td_error = (obtained_values - expected_values).detach().abs()
            self.memory.update_priorities(batch_indexes, td_error)

        ## Policy Training
        if self.policy_skip > 0:
            self.policy_skip -= 1
            return
        self.policy_skip = 4

        if self.target_value_net is not None:
            self.policy_net.calc_gradients(self.target_value_net, state_batch)
        else:
            self.policy_net.calc_gradients(self.value_net, state_batch)

        self.policy_net.step()

