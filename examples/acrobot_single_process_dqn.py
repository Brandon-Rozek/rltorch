import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import rltorch
import rltorch.network as rn
import rltorch.memory as M
import rltorch.env as E
from rltorch.action_selector import ArgMaxSelector
from rltorch.log import Logger

#
## Networks
#
class Value(nn.Module):
    def __init__(self, state_size, action_size):
        super(Value, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = rn.NoisyLinear(state_size, 255)
        self.fc_norm = nn.LayerNorm(255)
    
        self.value_fc = rn.NoisyLinear(255, 255)
        self.value_fc_norm = nn.LayerNorm(255)
        self.value = rn.NoisyLinear(255, 1)
    
        self.advantage_fc = rn.NoisyLinear(255, 255)
        self.advantage_fc_norm = nn.LayerNorm(255)
        self.advantage = rn.NoisyLinear(255, action_size)

    def forward(self, x):
        x = F.relu(self.fc_norm(self.fc1(x)))
    
        state_value = F.relu(self.value_fc_norm(self.value_fc(x)))
        state_value = self.value(state_value)
    
        advantage = F.relu(self.advantage_fc_norm(self.advantage_fc(x)))
        advantage = self.advantage(advantage)
    
        x = state_value + advantage - advantage.mean()
        return x

#
## Configuration
#
config = {}
config['seed'] = 901
config['environment_name'] = 'Acrobot-v1'
config['memory_size'] = 2000
config['total_training_episodes'] = 50
config['total_evaluation_episodes'] = 5
config['batch_size'] = 32
config['learning_rate'] = 1e-3
config['target_sync_tau'] = 1e-1
config['discount_rate'] = 0.99
config['replay_skip'] = 0
# How many episodes between printing out the episode stats
config['print_stat_n_eps'] = 1
config['disable_cuda'] = False
# Prioritized vs Random Sampling
# 0 - Random sampling
# 1 - Only the highest prioirities
config['prioritized_replay_sampling_priority'] = 0.6
# How important are the weights for the loss?
# 0 - Treat all losses equally
# 1 - Lower the importance of high losses
# Should ideally start from 0 and move your way to 1 to prevent overfitting
config['prioritized_replay_weight_importance'] = rltorch.scheduler.ExponentialScheduler(initial_value = 0.4, end_value = 1, iterations = 5000)

#
## Training Loop
#
def train(runner, agent, config, logwriter=None):
    finished = False
    last_episode_num = 1
    while not finished:
        runner.run(config['replay_skip'] + 1)
        agent.learn()
        if logwriter is not None:
          if last_episode_num < runner.episode_num:
                last_episode_num = runner.episode_num
                agent.net.log_named_parameters()
          logwriter.write(Logger)
        finished = runner.episode_num > config['total_training_episodes']


if __name__ == "__main__":
    # Setting up the environment
    rltorch.set_seed(config['seed'])
    print("Setting up environment...", end=" ")
    env = E.TorchWrap(gym.make(config['environment_name']))
    env.seed(config['seed'])
    print("Done.")
      
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Logging
    logwriter = None
    # logwriter = rltorch.log.LogWriter(SummaryWriter())

    # Setting up the networks
    device = torch.device("cuda:0" if torch.cuda.is_available() and not config['disable_cuda'] else "cpu")
    net = rn.Network(Value(state_size, action_size), 
                        torch.optim.Adam, config, device=device, name="DQN")
    target_net = rn.TargetNetwork(net, device=device)

    # Actor takes a net and uses it to produce actions from given states
    actor = ArgMaxSelector(net, action_size, device=device)
    # Memory stores experiences for later training
    memory = M.PrioritizedReplayMemory(capacity=config['memory_size'], alpha=config['prioritized_replay_sampling_priority'])

    # Runner performs a certain number of steps in the environment
    runner = rltorch.env.EnvironmentRunSync(env, actor, config, name="Training", memory=memory, logwriter=logwriter)

    # Agent is what performs the training
    agent = rltorch.agents.DQNAgent(net, memory, config, target_net=target_net)
    
    print("Training...")
    train(runner, agent, config, logwriter=logwriter) 

  # For profiling...
  # import cProfile
  # cProfile.run('train(runner, agent, config, logwriter = logwriter )')
  # python -m torch.utils.bottleneck /path/to/source/script.py [args] is also a good solution...

    print("Training Finished.")

    print("Evaluating...")
    rltorch.env.simulateEnvEps(env, actor, config, total_episodes = config['total_evaluation_episodes'], name = "Evaluation")
    print("Evaulations Done.")

    logwriter.close() # We don't need to write anything out to disk anymore
