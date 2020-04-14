from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

class A2CSingleAgent:
    def __init__(self, policy_net, value_net, memory, config, logger=None):
        self.policy_net = policy_net
        self.value_net = value_net
        self.memory = memory
        self.config = deepcopy(config)
        self.logger = logger

    def _discount_rewards(self, rewards):
        gammas = torch.ones_like(rewards)
        if len(rewards) > 1:
            discount_tensor = torch.tensor(self.config['discount_rate'])
            gammas[1:] = torch.cumprod(
                discount_tensor.repeat(len(rewards) - 1),
                dim=0
            )
        return gammas * rewards

    # This function is currently not used since the performance gains hasn't been shown
    # May be due to a faulty implementation, need to investigate more..
    def _generalized_advantage_estimation(self, states, rewards, next_states, not_done):
        tradeoff = 0.5
        with torch.no_grad():
            next_values = torch.zeros_like(rewards)
            next_values[not_done] = self.value_net(next_states[not_done]).squeeze(1)
            values = self.value_net(states).squeeze(1)

        generalized_advantages = torch.zeros_like(rewards)
        discount_tensor = torch.tensor(self.config['discount_rate']) * tradeoff
        for i, _ in enumerate(generalized_advantages):
            weights = torch.ones_like(rewards[i:])
            if i != len(generalized_advantages) - 1:
                weights[1:] = torch.cumprod(discount_tensor.repeat(len(rewards) - i - 1), dim=0)
            generalized_advantages[i] = (weights * (rewards[i:] + self.config['discount_rate'] * next_values[i:] - values[i:])).sum()

        return generalized_advantages

    def learn(self):
        episode_batch = self.memory.recall()
        state_batch, _, reward_batch, next_state_batch, done_batch, log_prob_batch = zip(*episode_batch)

        # Send batches to the appropriate device
        state_batch = torch.cat(state_batch).to(self.value_net.device)
        reward_batch = torch.tensor(reward_batch).to(self.value_net.device).float()
        not_done_batch = ~torch.tensor(done_batch).to(self.value_net.device)
        next_state_batch = torch.cat(next_state_batch).to(self.value_net.device)
        log_prob_batch = torch.cat(log_prob_batch).to(self.value_net.device)

        ## Value Loss
        # In A2C, the value loss is the difference between the discounted reward
        # and the value from the first state.
        # The value of the first state is supposed to tell us
        # the expected reward from the current policy of the whole episode
        discounted_reward = self._discount_rewards(reward_batch)
        observed_value = discounted_reward.sum()
        value_loss = F.mse_loss(observed_value, self.value_net(state_batch[0]))
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

        # advantages = self._generalized_advantage_estimation(state_batch, reward_batch, next_state_batch, not_done_batch)
        # Scale for more stable learning
        advantages = advantages / (advantages.std() + np.finfo('float').eps)

        policy_loss = (-log_prob_batch * advantages).sum()
    
        if self.logger is not None:
            self.logger.append("Loss/Policy", policy_loss.item())
            self.logger.append("Loss/Value", value_loss.item())

    
        self.policy_net.zero_grad()
        policy_loss.backward()
        self.policy_net.step()

        # Memory under the old policy is not needed for future training
        self.memory.clear()
