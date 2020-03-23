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
        
        if 'n_step' in self.config:
            batch_size = (self.config['batch_size'] // self.config['n_step']) * self.config['n_step']
            steps = self.config['n_step']
        else:
            batch_size = self.config['batch_size']
            steps = None
        
        if isinstance(self.memory, M.DQfDMemory):
            weight_importance = self.config['prioritized_replay_weight_importance']
            # If it's a scheduler then get the next value by calling next, otherwise just use it's value
            beta = next(weight_importance) if isinstance(weight_importance, collections.Iterable) else weight_importance

            # Check to see if we are doing N-Step DQN
            if steps is not None:
                minibatch = self.memory.sample_n_steps(batch_size, steps, beta)
            else:
                minibatch = self.memory.sample(batch_size, beta = beta)

            # Process batch
            state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, importance_weights, batch_indexes = M.zip_batch(minibatch, priority = True)

        else:
            # Check to see if we're doing N-Step DQN
            if steps is not None:
                minibatch = self.memory.sample_n_steps(batch_size, steps)
            else:
                minibatch = self.memory.sample(batch_size)

            # Process batch
            state_batch, action_batch, reward_batch, next_state_batch, not_done_batch, batch_indexes = M.zip_batch(minibatch, want_indices = True)

        batch_index_tensors = torch.tensor(batch_indexes)
        demo_mask = batch_index_tensors < self.memory.demo_position
        
        # Send to their appropriate devices
        state_batch = state_batch.to(self.net.device)
        action_batch = action_batch.to(self.net.device)
        reward_batch = reward_batch.to(self.net.device).float()
        next_state_batch = next_state_batch.to(self.net.device)
        not_done_batch = not_done_batch.to(self.net.device)

        state_values = self.net(state_batch)
        obtained_values = state_values.gather(1, action_batch.view(batch_size, 1))

        # DQN Loss
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

            best_next_state_value = torch.zeros(batch_size, device = self.net.device)
            best_next_state_value[not_done_batch] = next_state_values[not_done_batch].gather(1, next_best_action.view((not_done_size, 1))).squeeze(1)
            
        expected_values = (reward_batch + (self.config['discount_rate'] * best_next_state_value)).unsqueeze(1)

        # N-Step DQN Loss
        # num_steps capture how many steps actually exist before the end of episode
        if steps != None:
            expected_n_step_values = []
            with torch.no_grad():
                for i in range(0, len(state_batch), steps):
                    num_steps = not_done_batch[i:(i + steps)].sum()
                    if num_steps < 2:
                        continue # No point processing this
                    # Get the estimated value at the last state in a sequence
                    if self.target_net is not None:
                        expected_nth_values = self.target_net(state_batch[i + num_steps - 1].unsqueeze(0)).squeeze(0)
                        best_nth_action = self.net(state_batch[i + num_steps - 1].unsqueeze(0)).squeeze(0).argmax(0)
                    else:
                        expected_nth_values = self.net(state_batch[i + num_steps - 1].unsqueeze(0)).squeeze(0)
                        best_nth_action = expected_nth_values.argmax(0)
                    best_expected_nth_value = expected_nth_values[best_nth_action]
                    # Calculate the value leading up to it by taking the rewards and multiplying it by the discount rate
                    received_n_value = 0
                    for j in range(num_steps):
                        received_n_value += self.config['discount_rate']**j * reward_batch[j]
                    # Q(s, a) = r_0 + lambda_1 * r_1 + lambda_2^2 * r_2 + ... + lambda_{steps}^{steps} * max_{a}(Q(s + steps, a))
                    expected_n_step_values.append(received_n_value + self.config['discount_rate']**num_steps * best_expected_nth_value)
                expected_n_step_values = torch.stack(expected_n_step_values)
            # Gather the value the current network thinks it should be
            observed_n_step_values = []
            for i in range(0, len(state_batch), steps):
                num_steps = not_done_batch[i:(i + steps)].sum()
                if num_steps < 2:
                    continue # No point processing this
                observed_nth_value = self.net(state_batch[i].unsqueeze(0)).squeeze(0)[action_batch[i]]
                observed_n_step_values.append(observed_nth_value)
            observed_n_step_values = torch.stack(observed_n_step_values)

        # Demonstration loss
        if demo_mask.sum() > 0:
            l = torch.ones_like(state_values[demo_mask])
            expert_actions = action_batch[demo_mask]
            # l(s, a) is zero for every action the expert doesn't take
            for i,a in zip(range(len(l)), expert_actions):
                l[i].fill_(0.8) # According to paper
                l[i, a] = 0
            if self.target_net is not None:
                expert_value = self.target_net(state_batch[demo_mask])
            else:
                expert_value = state_values[demo_mask]
            expert_value = expert_value.gather(1, expert_actions.view(demo_mask.sum(), 1))
        
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
        if isinstance(self.memory, M.DQfDMemory):
            dqn_loss = (torch.as_tensor(importance_weights, device = self.net.device) * F.mse_loss(obtained_values, expected_values, reduction = 'none').squeeze(1)).mean()
        else:
            dqn_loss = F.mse_loss(obtained_values, expected_values)
        
        if steps != None:
            if isinstance(self.memory, M.DQfDMemory):
                dqn_n_step_loss =  (torch.as_tensor(importance_weights[::steps], device = self.net.device) * F.mse_loss(observed_n_step_values, expected_n_step_values, reduction = 'none')).mean()
            else:
                dqn_n_step_loss =  F.mse_loss(observed_n_step_values, expected_n_step_values, reduction = 'none').mean()
        else:
            dqn_n_step_loss = torch.tensor(0, device = self.net.device)
        
        if demo_mask.sum() > 0:
            if isinstance(self.memory, M.DQfDMemory):
                demo_loss = (torch.as_tensor(importance_weights, device = self.net.device)[demo_mask] * F.mse_loss((state_values[demo_mask] + l).max(1)[0].unsqueeze(1), expert_value, reduction = 'none').squeeze(1)).mean()
            else:
                demo_loss = F.mse_loss((state_values[demo_mask] + l).max(1)[0].unsqueeze(1), expert_value, reduction = 'none').squeeze(1).mean()
        else:
            demo_loss = 0.
        loss = td_importance * dqn_loss + td_importance * dqn_n_step_loss + demo_importance * demo_loss
        
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
        if isinstance(self.memory, M.DQfDMemory):
            td_error = (obtained_values - expected_values).detach().abs()
            td_error[demo_mask] = td_error[demo_mask] + self.config['demo_prio_bonus']
            observed_mask = batch_index_tensors >= self.memory.demo_position
            td_error[observed_mask] = td_error[observed_mask] + self.config['observed_prio_bonus']
            self.memory.update_priorities(batch_indexes, td_error)


