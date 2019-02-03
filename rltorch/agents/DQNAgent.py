import rltorch.memory as M
import torch
import torch.nn.functional as F
from copy import deepcopy

class DQNAgent:
    def __init__(self, net , memory, config, target_net = None, logger = None):
        self.net = net
        self.target_net = target_net
        self.memory = memory
        self.config = deepcopy(config)
        self.logger = logger

    def learn(self):
        if len(self.memory) < self.config['batch_size']:
            return

        minibatch = self.memory.sample(self.config['batch_size'])
        state_batch, action_batch, reward_batch, next_state_batch, not_done_batch = M.zip_batch(minibatch)
        
        # Send to their appropriate devices
        state_batch = state_batch.to(self.net.device)
        action_batch = action_batch.to(self.net.device)
        reward_batch = reward_batch.to(self.net.device)
        next_state_batch = next_state_batch.to(self.net.device)
        not_done_batch = not_done_batch.to(self.net.device)

        obtained_values = self.net(state_batch).gather(1, action_batch.view(self.config['batch_size'], 1))

        with torch.no_grad():
            # Use the target net to produce action values for the next state
            # and the regular net to select the action
            # That way we decouple the value and action selecting processes (DOUBLE DQN)
            not_done_size = not_done_batch.sum()
            if self.target_net is not None:
                next_state_values = self.target_net(next_state_batch)
                next_best_action = self.net(next_state_batch).argmax(1)
            else:
                next_state_values = self.net(next_state_batch)
                next_best_action = next_state_values.argmax(1)

            best_next_state_value = torch.zeros(self.config['batch_size'])
            best_next_state_value[not_done_batch] = next_state_values.gather(1, next_best_action.view((not_done_size, 1))).squeeze(1)
            
        expected_values = (reward_batch + (self.config['discount_rate'] * best_next_state_value)).unsqueeze(1)

        loss = F.mse_loss(obtained_values, expected_values)
        
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
