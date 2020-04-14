import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import rltorch
import rltorch.network as rn
import rltorch.memory as M
import rltorch.env as E
from rltorch.action_selector import ArgMaxSelector
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
from rltorch.log import Logger

#
## Networks
#
class Value(nn.Module):
    def __init__(self, state_size, action_size):
        super(Value, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
    
        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv_norm1 = nn.LayerNorm([32, 19, 19])
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))    
        self.conv_norm2 = nn.LayerNorm([64, 8, 8])
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.conv_norm3 = nn.LayerNorm([64, 6, 6])
    
        self.fc1 = rn.NoisyLinear(64 * 6 * 6, 384)
        self.fc_norm = nn.LayerNorm(384)
    
        self.value_fc = rn.NoisyLinear(384, 384)
        self.value_fc_norm = nn.LayerNorm(384)
        self.value = rn.NoisyLinear(384, 1)
    
        self.advantage_fc = rn.NoisyLinear(384, 384)
        self.advantage_fc_norm = nn.LayerNorm(384)
        self.advantage = rn.NoisyLinear(384, action_size)

  
    def forward(self, x):
        x = F.relu(self.conv_norm1(self.conv1(x)))
        x = F.relu(self.conv_norm2(self.conv2(x)))
        x = F.relu(self.conv_norm3(self.conv3(x)))
    
        # Makes batch_size dimension again
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc_norm(self.fc1(x)))
    
        state_value = F.relu(self.value_fc_norm(self.value_fc(x)))
        state_value = self.value(state_value)
    
        advantage = F.relu(self.advantage_fc_norm(self.advantage_fc(x)))
        advantage = self.advantage(advantage)
    
        x = state_value + advantage - advantage.mean()
    
        # For debugging purposes...
        if torch.isnan(x).any().item():
            print("WARNING NAN IN MODEL DETECTED")
    
        return x

#
## Configuration
#
config = {}
config['seed'] = 901
config['environment_name'] = 'PongNoFrameskip-v4'
config['memory_size'] = 5000
config['total_training_episodes'] = 500
config['total_evaluation_episodes'] = 10
config['learning_rate'] = 1e-4
config['target_sync_tau'] = 1e-3
config['discount_rate'] = 0.99
config['exploration_rate'] = rltorch.scheduler.ExponentialScheduler(initial_value = 0.1, end_value = 0.01, iterations = 5000)
config['replay_skip'] = 4
config['batch_size'] = 32 * (config['replay_skip'] + 1)
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
def train(runner, agent, config, logwriter = None):
    finished = False
    while not finished:
        runner.run()
        agent.learn()
        if logwriter is not None:
            agent.value_net.log_named_parameters()
            agent.policy_net.log_named_parameters()
            logwriter.write(Logger)
        finished = runner.episode_num > config['total_training_episodes']


if __name__ == "__main__":
    # To not hit file descriptor memory limit
    torch.multiprocessing.set_sharing_strategy('file_system') 

    # Setting up the environment
    rltorch.set_seed(config['seed'])
    print("Setting up environment...", end = " ")
    env = E.FrameStack(E.TorchWrap(
        E.ProcessFrame(E.FireResetEnv(gym.make(config['environment_name'])), 
        resize_shape=(80, 80), crop_bounds=[34, 194, 15, 145], grayscale=True))
    , 4)
    env.seed(config['seed'])
    print("Done.")
      
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Logging
    logwriter = rltorch.log.LogWriter(SummaryWriter())

    # Setting up the networks
    device = torch.device("cuda:0" if torch.cuda.is_available() and not config['disable_cuda'] else "cpu")
    net = rn.Network(Value(state_size, action_size), 
        torch.optim.Adam, config, device=device, name="DQN")
    target_net = rn.TargetNetwork(net, device=device)
    net.model.share_memory()
    target_net.model.share_memory()

    # Actor takes a net and uses it to produce actions from given states
    actor = ArgMaxSelector(net, action_size, device=device)
    # Memory stores experiences for later training
    memory = M.PrioritizedReplayMemory(capacity=config['memory_size'], alpha=config['prioritized_replay_sampling_priority'])

    # Runner performs a certain number of steps in the environment
    runner = rltorch.mp.EnvironmentRun(env, actor, config, name="Training", memory=memory, logwriter=logwriter)

    # Agent is what performs the training
    agent = rltorch.agents.DQNAgent(net, memory, config, target_net=target_net)
    
    print("Training...")
    train(runner, agent, config, logwriter=logwriter)

  # For profiling...
  # import cProfile
  # cProfile.run('train(runner, agent, config, logwriter = logwriter )')
  # python -m torch.utils.bottleneck /path/to/source/script.py [args] is also a good solution...

    print("Training Finished.")
    runner.terminate() # We don't need the extra process anymore

    print("Evaluating...")
    rltorch.env.simulateEnvEps(env, actor, config, total_episodes=config['total_evaluation_episodes'], name="Evaluation")
    print("Evaulations Done.")

    logwriter.close() # We don't need to write anything out to disk anymore
