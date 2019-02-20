import rltorch
from copy import deepcopy
import torch
import numpy as np

class REINFORCEAgent:
  def __init__(self, net , memory, config, target_net = None, logger = None):
    self.net = net
    if not isinstance(memory, rltorch.memory.EpisodeMemory):
      raise ValueError("Memory must be of instance EpisodeMemory")
    self.memory = memory
    self.config = deepcopy(config)
    self.target_net = target_net
    self.logger = logger

  def _discount_rewards(self, rewards):
    discounted_rewards = torch.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(len(rewards))):
      running_add = running_add * self.config['discount_rate'] + rewards[t]
      discounted_rewards[t] = running_add

    # Normalize rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + np.finfo('float').eps)
    return discounted_rewards
  
  def learn(self):
    episode_batch = self.memory.recall()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_prob_batch = zip(*episode_batch)

    discount_reward_batch = self._discount_rewards(torch.tensor(reward_batch))
    log_prob_batch = torch.cat(log_prob_batch)

    policy_loss = (-log_prob_batch * discount_reward_batch).sum()
    
    if self.logger is not None:
            self.logger.append("Loss", policy_loss.item())

    self.net.zero_grad()
    policy_loss.backward()
    self.net.clamp_gradients()
    self.net.step()

    if self.target_net is not None:
      if 'target_sync_tau' in self.config:
        self.target_net.partial_sync(self.config['target_sync_tau'])
      else:
        self.target_net.sync()

    # Memory is irrelevant for future training
    self.memory.clear()
