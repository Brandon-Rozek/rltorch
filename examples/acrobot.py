import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rltorch
import rltorch.network as rn
import rltorch.memory as M
import rltorch.env as E
from rltorch.action_selector import ArgMaxSelector
from tensorboardX import SummaryWriter


class Value(nn.Module):
  def __init__(self, state_size, action_size):
    super(Value, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    
    self.fc1 = rn.NoisyLinear(state_size, 64)
    
    self.value_fc = rn.NoisyLinear(64, 64)
    self.value = rn.NoisyLinear(64, 1)
    
    self.advantage_fc = rn.NoisyLinear(64, 64)
    self.advantage = rn.NoisyLinear(64, action_size)

  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    
    state_value = F.relu(self.value_fc(x))
    state_value = self.value(state_value)
    
    advantage = F.relu(self.advantage_fc(x))
    advantage = self.advantage(advantage)
    
    x = state_value + advantage - advantage.mean()
    
    return x


config = {}
config['seed'] = 901
config['environment_name'] = 'Acrobot-v1'
config['memory_size'] = 2000
config['total_training_episodes'] = 50
config['total_evaluation_episodes'] = 10
config['batch_size'] = 32
config['learning_rate'] = 1e-3
config['target_sync_tau'] = 1e-1
config['weight_decay'] = 1e-5
config['discount_rate'] = 0.99
config['replay_skip'] = 0
# How many episodes between printing out the episode stats
config['print_stat_n_eps'] = 1
config['disable_cuda'] = False

def train(runner, agent, config, logwriter = None):
    finished = False
    episode_num = 1
    while not finished:
        runner.run(config['replay_skip'] + 1, printstat = runner.episode_num % config['print_stat_n_eps'] == 0)
        agent.learn()
        runner.join()
        # When the episode number changes, write out the weight histograms
        if logwriter is not None and episode_num < runner.episode_num:
            episode_num = runner.episode_num
            agent.net.log_named_parameters()
        
        if logwriter is not None:
            logwriter.write()
        finished = runner.episode_num > config['total_training_episodes']


# Setting up the environment
rltorch.set_seed(config['seed'])
print("Setting up environment...", end = " ")
env = E.TorchWrap(gym.make(config['environment_name']))
env.seed(config['seed'])
print("Done.")
    
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Logging
logger = rltorch.log.Logger()
logwriter = rltorch.log.LogWriter(logger, SummaryWriter())

# Setting up the networks
device = torch.device("cuda:0" if torch.cuda.is_available() and not config['disable_cuda'] else "cpu")
net = rn.Network(Value(state_size, action_size), 
                    torch.optim.Adam, config, device = device, logger = logger, name = "DQN")
target_net = rn.TargetNetwork(net, device = device)

# Actor takes a net and uses it to produce actions from given states
actor = ArgMaxSelector(net, action_size, device = device)
# Memory stores experiences for later training
memory = M.ReplayMemory(capacity = config['memory_size'])

# Runner performs a certain number of steps in the environment
runner = rltorch.mp.EnvironmentRun(env, actor, config, memory = memory, logger = logger, name = "Training")
runner.start()

# Agent is what performs the training
agent = rltorch.agents.DQNAgent(net, memory, config, target_net = target_net, logger = logger)
   
print("Training...")
train(runner, agent, config, logwriter = logwriter) 

# For profiling...
# import cProfile
# cProfile.run('train(runner, agent, config, logwriter = logwriter )')
# python -m torch.utils.bottleneck /path/to/source/script.py [args] is also a good solution...

print("Training Finished.")
runner.terminate() # We don't need the extra process anymore

print("Evaluating...")
rltorch.env.simulateEnvEps(env, actor, config, total_episodes = config['total_evaluation_episodes'], logger = logger, name = "Evaluation")
print("Evaulations Done.")

logwriter.close() # We don't need to write anything out to disk anymore