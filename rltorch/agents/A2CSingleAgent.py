from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import rltorch
import rltorch.memory as M
import collections
import random

class A2CSingleAgent:
  def __init__(self, policy_net, value_net, memory, config, logger = None):
    self.policy_net = policy_net
    self.value_net = value_net
    self.memory = memory
    self.config = deepcopy(config)
    self.logger = logger

  def _discount_rewards(self, rewards):
    discounted_rewards = torch.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
      running_add = running_add * self.config['discount_rate'] + rewards[t]
      discounted_rewards[t] = running_add

    return discounted_rewards
  
  
  def learn(self):
    if len(self.memory) < self.config['batch_size']:
      return

    episode_batch = self.memory.recall()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_prob_batch = zip(*episode_batch)  

    state_batch = torch.cat(state_batch).to(self.value_net.device)
    reward_batch = torch.tensor(reward_batch).to(self.value_net.device)
    not_done_batch = ~torch.tensor(done_batch).to(self.value_net.device)
    next_state_batch = torch.cat(next_state_batch).to(self.value_net.device)
    log_prob_batch = torch.cat(log_prob_batch).to(self.value_net.device)

    ## Value Loss
    value_loss = F.mse_loss(self._discount_rewards(reward_batch), self.value_net(state_batch[0]))
    self.value_net.zero_grad()
    value_loss.backward()
    self.value_net.step()

    ## Policy Loss
    with torch.no_grad():
      state_values = self.value_net(state_batch)
      next_state_values = torch.zeros_like(state_values) 
      next_state_values[not_done_batch] = self.value_net(next_state_batch[not_done_batch])
    advantages = (reward_batch.unsqueeze(1) + self.config['discount_rate'] * next_state_values) - state_values
    advantages = advantages.squeeze(1)

    policy_loss = (-log_prob_batch * advantages).sum()
    
    if self.logger is not None:
      self.logger.append("Loss/Policy", policy_loss.item())
      self.logger.append("Loss/Value", value_loss.item())

    
    self.policy_net.zero_grad()
    policy_loss.backward()
    self.policy_net.step()


    # Memory is irrelevant for future training
    self.memory.clear()
    

