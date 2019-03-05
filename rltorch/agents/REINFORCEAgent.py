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

  # Shaped rewards implements three improvements to REINFORCE
  # 1) Discounted rewards, future rewards matter less than current
  # 2) Baselines: We use the mean reward to see if the current reward is advantageous or not
  # 3) Causality: Your current actions do not affect your past. Only the present and future.
  def _shape_rewards(self, rewards):
    shaped_rewards = torch.zeros_like(rewards)
    baseline = rewards.mean()
    for i in range(len(rewards)):
      gammas = torch.ones_like(rewards[i:])
      if i != len(rewards) - 1:
        gammas[1:] = torch.cumprod(torch.tensor(self.config['discount_rate']).repeat(len(rewards) - i - 1), dim = 0)
      advantages = rewards[i:] - baseline
      shaped_rewards[i] = (gammas * advantages).sum()
    return shaped_rewards
  
  def learn(self):
    episode_batch = self.memory.recall()
    state_batch, action_batch, reward_batch, next_state_batch, done_batch, log_prob_batch = zip(*episode_batch)

    # Caluclate discounted rewards to place more importance to recent rewards
    shaped_reward_batch = self._shape_rewards(torch.tensor(reward_batch))

    # Scale discounted rewards to have variance 1 (stabalizes training)
    shaped_reward_batch = shaped_reward_batch / (shaped_reward_batch.std() + np.finfo('float').eps)

    log_prob_batch = torch.cat(log_prob_batch)

    policy_loss = (-log_prob_batch * shaped_reward_batch).sum()

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

    # Memory under the old policy is not needed for future training
    self.memory.clear()
