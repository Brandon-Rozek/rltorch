from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import rltorch
import rltorch.memory as M
import collections
import random

class PPOAgent:
  def __init__(self, policy_net, value_net, memory, config, logger = None):
    self.policy_net = policy_net
    self.old_policy_net = rltorch.network.TargetNetwork(policy_net)
    self.value_net = value_net
    self.memory = memory
    self.config = deepcopy(config)
    self.logger = logger

  def _discount_rewards(self, rewards):
    gammas = torch.ones_like(rewards)
    if len(rewards) > 1:
      gammas[1:] = torch.cumprod(torch.tensor(self.config['discount_rate']).repeat(len(rewards) - 1), dim = 0)
    return gammas * rewards
  
  
  def learn(self):
    episode_batch = self.memory.recall()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_prob_batch = zip(*episode_batch)  

    # Send batches to the appropriate device
    state_batch = torch.cat(state_batch).to(self.value_net.device)
    action_batch = torch.tensor(action_batch).to(self.value_net.device)
    reward_batch = torch.tensor(reward_batch).to(self.value_net.device)
    not_done_batch = ~torch.tensor(done_batch).to(self.value_net.device)
    next_state_batch = torch.cat(next_state_batch).to(self.value_net.device)
    log_prob_batch = torch.cat(log_prob_batch).to(self.value_net.device)

    ## Value Loss
    # In PPO, the value loss is the difference between the discounted reward and the value from the first state
    # The value of the first state is supposed to tell us the expected reward from the current policy of the whole episode
    value_loss = F.mse_loss(self._discount_rewards(reward_batch).sum(), self.value_net(state_batch[0]))
    self.value_net.zero_grad()
    value_loss.backward()
    self.value_net.step()

    ## Policy Loss
    # Increase probabilities of advantageous states 
    # and decrease the probabilities of non-advantageous ones
    with torch.no_grad():
      state_values = self.value_net(state_batch)
      next_state_values = torch.zeros_like(state_values) 
      next_state_values[not_done_batch] = self.value_net(next_state_batch[not_done_batch])
    advantages = (reward_batch.unsqueeze(1) + self.config['discount_rate'] * next_state_values) - state_values
    advantages = advantages.squeeze(1)

    action_probabilities = self.old_policy_net(state_batch).detach()
    distributions = list(map(Categorical, action_probabilities))
    old_log_probs = torch.stack(list(map(lambda distribution, action: distribution.log_prob(action), distributions, action_batch)))

    # For PPO we want to stay within a certain ratio of the old policy
    policy_ratio = torch.exp(log_prob_batch - old_log_probs) # Equivalent to (log_prob / old_log_prob)
    policy_loss1 = policy_ratio * advantages 
    policy_loss2 = policy_ratio.clamp(min = 0.8, max = 1.2) * advantages # From original paper
    policy_loss = -torch.min(policy_loss1, policy_loss2).sum()
    
    if self.logger is not None:
      self.logger.append("Loss/Policy", policy_loss.item())
      self.logger.append("Loss/Value", value_loss.item())

    
    self.old_policy_net.sync()
    self.policy_net.zero_grad()
    policy_loss.backward()
    self.policy_net.step()


    # Memory under the old policy is not needed for future training
    self.memory.clear()
    

