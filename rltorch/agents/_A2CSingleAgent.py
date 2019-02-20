# Deprecated since the idea of the idea shouldn't work without having some sort of "mental model" of the environment

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
  def __init__(self, policy_net, value_net, memory, config, target_value_net = None, logger = None):
    self.policy_net = policy_net
    self.value_net = value_net
    self.memory = memory
    self.config = deepcopy(config)
    self.target_value_net = target_value_net
    self.logger = logger

  def learn_value(self):
    if (isinstance(self.memory, M.PrioritizedReplayMemory)):
      weight_importance = self.config['prioritized_replay_weight_importance']
      # If it's a scheduler then get the next value by calling next, otherwise just use it's value
      beta = next(weight_importance) if isinstance(weight_importance, collections.Iterable) else weight_importance
      minibatch = self.memory.sample(self.config['batch_size'], beta = beta)
      state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, importance_weights, batch_indexes = M.zip_batch(minibatch, priority = True)
    else:
      minibatch = self.memory.sample(self.config['batch_size'])
      state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = M.zip_batch(minibatch)
    
    # Send to their appropriate devices 
    state_batch = state_batch.to(self.value_net.device)
    action_batch = action_batch.to(self.value_net.device)
    reward_batch = reward_batch.to(self.value_net.device)
    next_state_batch = next_state_batch.to(self.value_net.device)
    not_done_batch = not_done_batch.to(self.value_net.device)


    ## Value Loss
    state_values = self.value_net(state_batch)
    obtained_values = state_values.gather(1, action_batch.view(self.config['batch_size'], 1))
    with torch.no_grad():
      # Use the target net to produce action values for the next state
      # and the regular net to select the action
      # That way we decouple the value and action selecting processes (DOUBLE DQN)
      not_done_size = not_done_batch.sum()
      next_state_values = torch.zeros_like(state_values)
      if self.target_value_net is not None:
        next_state_values[not_done_batch] = self.target_value_net(next_state_batch[not_done_batch])
        next_best_action = self.value_net(next_state_batch).argmax(1)
      else:
        next_state_values[not_done_batch] = self.value_net(next_state_batch[not_done_batch])
        next_best_action = next_state_values.argmax(1)

      best_next_state_value = torch.zeros(self.config['batch_size'], device = self.value_net.device)
      # best_next_state_value[not_done_batch] = next_state_values.gather(1, next_best_action.view((not_done_size, 1))).squeeze(1)
      best_next_state_value[not_done_batch] = next_state_values[not_done_batch].gather(1, next_best_action[not_done_batch].view((not_done_size, 1))).squeeze(1)
            
    expected_values = (reward_batch + (self.config['discount_rate'] * best_next_state_value)).unsqueeze(1)

    if (isinstance(self.memory, M.PrioritizedReplayMemory)):
      importance_weights = torch.as_tensor(importance_weights, device = self.value_net.device) 
      value_loss = (importance_weights * ((obtained_values - expected_values)**2).squeeze(1)).mean()
    else:
      value_loss = F.mse_loss(obtained_values, expected_values)

    if (isinstance(self.memory, M.PrioritizedReplayMemory)):
      td_error = (obtained_values - expected_values).detach().abs()
      self.memory.update_priorities(batch_indexes, td_error)

    self.value_net.zero_grad()
    value_loss.backward()
    self.value_net.step()
    
    if self.target_value_net is not None:
      if 'target_sync_tau' in self.config:
        self.target_value_net.partial_sync(self.config['target_sync_tau'])
      else:
        self.target_value_net.sync()

    if self.logger is not None:
      self.logger.append("Loss/Value", value_loss.item())
    

  def learn_policy(self):
    starting_index = random.randint(0, len(self.memory) - self.config['batch_size'])
    state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = M.zip_batch(self.memory[starting_index:(starting_index + self.config['batch_size'])])
    
    state_batch = state_batch.to(self.policy_net.device)
    action_batch = action_batch.to(self.policy_net.device)
    reward_batch = reward_batch.to(self.policy_net.device)
    next_state_batch = next_state_batch.to(self.policy_net.device)
    not_done_batch = not_done_batch.to(self.policy_net.device)

    # Find when episode ends and filter out the Transitions after
    episode_ends = (~not_done_batch).nonzero().squeeze(1)
    start_idx = 0
    end_idx = self.config['batch_size']
    if len(episode_ends) > 0:
      if (episode_ends[0] == 0).item():
        if len(episode_ends) > 1:
          start_idx = 1
          end_idx = episode_ends[1] + 1
        else:
          start_idx = 1
      else:
        end_idx = episode_ends[0] + 1
    batch_size = end_idx - start_idx

    # Now filter...
    state_batch = state_batch[start_idx:end_idx]
    action_batch = action_batch[start_idx:end_idx]
    reward_batch = reward_batch[start_idx:end_idx]
    next_state_batch = next_state_batch[start_idx:end_idx]
    not_done_batch = not_done_batch[start_idx:end_idx]


    with torch.no_grad():
      if self.target_value_net is not None:
        state_values = self.target_value_net(state_batch)
        next_state_values = torch.zeros_like(state_values, device = self.value_net.device) 
        next_state_values[not_done_batch] = self.target_value_net(next_state_batch[not_done_batch])
      else:
        state_values = self.value_net(state_batch)
        next_state_values = torch.zeros_like(state_values, device = self.value_net.device) 
        next_state_values[not_done_batch] = self.value_net(next_state_batch[not_done_batch])
      
    obtained_values = state_values.gather(1, action_batch.view(batch_size, 1))
    approx_state_action_values = reward_batch.unsqueeze(1) + self.config['discount_rate'] * next_state_values
    advantage = (obtained_values - approx_state_action_values.mean(1).unsqueeze(1)) 
    # Scale and squeeze the dimension
    advantage = advantage.squeeze(1)
    # advantage = (advantage / (state_values.std() + np.finfo('float').eps)).squeeze(1)
    action_probabilities = self.policy_net(state_batch)
    distributions = list(map(Categorical, action_probabilities))
    log_probs = torch.stack(list(map(lambda distribution, action: distribution.log_prob(action), distributions, action_batch)))
    policy_loss = (-log_probs * advantage).mean()

    self.policy_net.zero_grad()
    policy_loss.backward()
    self.policy_net.step()

    if self.logger is not None:
      self.logger.append("Loss/Policy", policy_loss.item())
  
  def learn(self):
    if len(self.memory) < self.config['batch_size']:
      return
    self.learn_value()
    self.learn_policy()
    

    




  # def learn(self):
  #   if len(self.memory) < self.config['batch_size']:
  #     return
    
  #   if (isinstance(self.memory, M.PrioritizedReplayMemory)):
  #     weight_importance = self.config['prioritized_replay_weight_importance']
  #     # If it's a scheduler then get the next value by calling next, otherwise just use it's value
  #     beta = next(weight_importance) if isinstance(weight_importance, collections.Iterable) else weight_importance
  #     minibatch = self.memory.sample(self.config['batch_size'], beta = beta)
  #     state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, importance_weights, batch_indexes = M.zip_batch(minibatch, priority = True)
  #   else:
  #     minibatch = self.memory.sample(self.config['batch_size'])
  #     state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = M.zip_batch(minibatch)
    
  #   # Send to their appropriate devices 
  #   # [TODO] Notice how we're sending it to the value_net's device, what if policy_net was on a different device?
  #   state_batch = state_batch.to(self.value_net.device)
  #   action_batch = action_batch.to(self.value_net.device)
  #   reward_batch = reward_batch.to(self.value_net.device)
  #   next_state_batch = next_state_batch.to(self.value_net.device)
  #   not_done_batch = not_done_batch.to(self.value_net.device)


  #   ## Value Loss

  #   obtained_values = self.value_net(state_batch).gather(1, action_batch.view(self.config['batch_size'], 1))

  #   with torch.no_grad():
  #     # Use the target net to produce action values for the next state
  #     # and the regular net to select the action
  #     # That way we decouple the value and action selecting processes (DOUBLE DQN)
  #     not_done_size = not_done_batch.sum()
  #     if self.target_value_net is not None:
  #       next_state_values = self.target_value_net(next_state_batch)
  #       next_best_action = self.value_net(next_state_batch).argmax(1)
  #     else:
  #       next_state_values = self.value_net(next_state_batch)
  #       next_best_action = next_state_values.argmax(1)

  #     best_next_state_value = torch.zeros(self.config['batch_size'], device = self.value_net.device)
  #     best_next_state_value[not_done_batch] = next_state_values.gather(1, next_best_action.view((not_done_size, 1))).squeeze(1)
            
  #   expected_values = (reward_batch + (self.config['discount_rate'] * best_next_state_value)).unsqueeze(1)

  #   if (isinstance(self.memory, M.PrioritizedReplayMemory)):
  #     importance_weights = torch.as_tensor(importance_weights, device = self.value_net.device) 
  #     value_loss = (importance_weights * ((obtained_values - expected_values)**2).squeeze(1)).mean()
  #   else:
  #     value_loss = F.mse_loss(obtained_values, expected_values)
    
  #   self.value_net.zero_grad()
  #   value_loss.backward()
  #   self.value_net.step()

  #   if self.target_value_net is not None:
  #     if 'target_sync_tau' in self.config:
  #       self.target_value_net.partial_sync(self.config['target_sync_tau'])
  #     else:
  #       self.target_value_net.sync()

  #   if (isinstance(self.memory, M.PrioritizedReplayMemory)):
  #     td_error = (obtained_values - expected_values).detach().abs()
  #     self.memory.update_priorities(batch_indexes, td_error)

  #   if self.logger is not None:
  #     self.logger.append("ValueLoss", value_loss.item())

  #   ## Policy Loss
  #   with torch.no_grad():
  #     state_values = self.value_net(state_batch)
  #     if self.target_value_net is not None:
  #       next_state_values = self.target_value_net(next_state_batch)
  #     else:
  #       next_state_values = self.value_net(next_state_batch)
      
  #   state_action_values = state_values.gather(1, action_batch.view(self.config['batch_size'], 1))
  #   average_next_state_values = torch.zeros(self.config['batch_size'], device = self.value_net.device) 
  #   average_next_state_values[not_done_batch] = next_state_values.mean(1)

  #   advantage = (state_action_values - (reward_batch + self.config['discount_rate'] * average_next_state_values).unsqueeze(1)) 
  #   # Scale and squeeze the dimension
  #   advantage = advantage.squeeze(1)
  #   # advantage = (advantage / (state_values.std() + np.finfo('float').eps)).squeeze(1)
  #   action_probabilities = self.policy_net(state_batch)
  #   distributions = list(map(Categorical, action_probabilities))
  #   log_probs = torch.stack(list(map(lambda distribution, action: distribution.log_prob(action), distributions, action_batch)))
  #   if (isinstance(self.memory, M.PrioritizedReplayMemory)):
  #     policy_loss = (importance_weights * -log_probs * advantage).sum()
  #   else:
  #     policy_loss = (-log_probs * advantage).sum()

  #   self.policy_net.zero_grad()
  #   policy_loss.backward()
  #   self.policy_net.step()

  #   if self.logger is not None:
  #     self.logger.append("PolicyLoss", policy_loss.item())




