import collections
import rltorch.memory as M
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from pathlib import Path
from rltorch.action_selector import ArgMaxSelector

class DQfDAgent:
    def __init__(self, net, memory, config, target_net = None, logger = None):
        self.net = net
        self.target_net = target_net
        self.memory = memory
        self.config = deepcopy(config)
        self.logger = logger
    def save(self, file_location):
        torch.save(self.net.model.state_dict(), file_location)
    def load(self, file_location):
        self.net.model.state_dict(torch.load(file_location))
        self.net.model.to(self.net.device)
        self.target_net.sync()
    
    def learn(self, logger = None):
        if len(self.memory) < self.config['batch_size']:
            return
        
        weight_importance = self.config['prioritized_replay_weight_importance']
        # If it's a scheduler then get the next value by calling next, otherwise just use it's value
        beta = next(weight_importance) if isinstance(weight_importance, collections.Iterable) else weight_importance
        minibatch = self.memory.sample(self.config['batch_size'], beta = beta)
        state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, importance_weights, batch_indexes = M.zip_batch(minibatch, priority = True)
        
        demo_indexes = batch_indexes < self.memory.demo_position
        
        # Send to their appropriate devices
        state_batch = state_batch.to(self.net.device)
        action_batch = action_batch.to(self.net.device)
        reward_batch = reward_batch.to(self.net.device).float()
        next_state_batch = next_state_batch.to(self.net.device)
        not_done_batch = not_done_batch.to(self.net.device)

        state_values = self.net(state_batch)
        obtained_values = state_values.gather(1, action_batch.view(self.config['batch_size'], 1))

        with torch.no_grad():
            # Use the target net to produce action values for the next state
            # and the regular net to select the action
            # That way we decouple the value and action selecting processes (DOUBLE DQN)
            not_done_size = not_done_batch.sum()
            next_state_values = torch.zeros_like(state_values, device = self.net.device)
            if self.target_net is not None:
                next_state_values[not_done_batch] = self.target_net(next_state_batch[not_done_batch])
                next_best_action = self.net(next_state_batch[not_done_batch]).argmax(1)
            else:
                next_state_values[not_done_batch] = self.net(next_state_batch[not_done_batch])
                next_best_action = next_state_values[not_done_batch].argmax(1)

            best_next_state_value = torch.zeros(self.config['batch_size'], device = self.net.device)
            best_next_state_value[not_done_batch] = next_state_values[not_done_batch].gather(1, next_best_action.view((not_done_size, 1))).squeeze(1)
            
        expected_values = (reward_batch + (self.config['discount_rate'] * best_next_state_value)).unsqueeze(1)

        # Demonstration loss
        l = torch.ones_like(state_values[demo_indexes])
        expert_actions = action_batch[demo_indexes]
        # l(s, a) is zero for every action the expert doesn't take
        for i,a in zip(range(len(l)), expert_actions):
            l[i].fill_(0.8) # According to paper
            l[i, a] = 0
        if self.target_net is not None:
            expert_value = self.target_net(state_batch[demo_indexes])
        else:
            expert_value = state_values[demo_indexes]
        expert_value = expert_value.gather(1, expert_actions.view((self.config['batch_size'], 1))).squeeze(1)

        # Iterate through hyperparamters
        if isinstance(self.config['dqfd_demo_loss_weight'], collections.Iterable):
            demo_importance = next(self.config['dqfd_demo_loss_weight'])
        else: 
            demo_importance = self.config['dqfd_demo_loss_weight']
        if isinstance(self.config['dqfd_td_loss_weight'], collections.Iterable):
            td_importance = next(self.config['dqfd_td_loss_weight'])
        else: 
            td_importance = self.config['dqfd_td_loss_weight']
        
        # Since dqn_loss and demo_loss are different sizes, the reduction has to happen before they are combined
        dqn_loss = (torch.as_tensor(importance_weights, device = self.net.device) * F.mse_loss(obtained_values, expected_values, reduction = 'none')).mean()
        demo_loss = (torch.as_tensor(importance_weights[demo_indexes], device = self.net.device) * F.mse_loss((state_values[demo_indexes] + l).max(1)[0], expert_value, reduction = 'none')).mean()
        loss = td_importance * dqn_loss + demo_importance * demo_loss
        
        if self.logger is not None:
            self.logger.append("Loss", loss.item())

        self.net.zero_grad()
        loss.backward()
        self.net.clamp_gradients()
        self.net.step()

        if self.target_net is not None:
            if 'target_sync_tau' in self.config:
                self.target_net.partial_sync(self.config['target_sync_tau'])
            else:
                self.target_net.sync()
        
        # If we're sampling by TD error, readjust the weights of the experiences
        # TODO: Can probably adjust demonstration priority here
        td_error = (obtained_values - expected_values).detach().abs()
        self.memory.update_priorities(batch_indexes, td_error)


