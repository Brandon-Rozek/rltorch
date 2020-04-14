import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import rltorch
import rltorch.network as rn
import rltorch.memory as M
import rltorch.env as E
from rltorch.action_selector import StochasticSelector
from tensorboardX import SummaryWriter
from rltorch.log import Logger

#
## Networks
#
class Value(nn.Module):
    def __init__(self, state_size):
        super(Value, self).__init__()
        self.state_size = state_size

        self.fc1 = rn.NoisyLinear(state_size, 64)
        self.fc_norm = nn.LayerNorm(64)

        self.fc2 = rn.NoisyLinear(64, 64)
        self.fc2_norm = nn.LayerNorm(64)

        self.fc3 = rn.NoisyLinear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc_norm(self.fc1(x)))
        x = F.relu(self.fc2_norm(self.fc2(x)))
        x = self.fc3(x)
        return x

class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = rn.NoisyLinear(state_size, 64)
        self.fc_norm = nn.LayerNorm(64)

        self.fc2 = rn.NoisyLinear(64, 64)
        self.fc2_norm = nn.LayerNorm(64)

        self.fc3 = rn.NoisyLinear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc_norm(self.fc1(x)))
        x = F.relu(self.fc2_norm(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim = 1)
        return x

#
## Configuration
#
config = {}
config['seed'] = 901
config['environment_name'] = 'Acrobot-v1'
config['total_training_episodes'] = 500
config['total_evaluation_episodes'] = 10
config['learning_rate'] = 1e-3
config['discount_rate'] = 0.99
# How many episodes between printing out the episode stats
config['print_stat_n_eps'] = 1
config['disable_cuda'] = False

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
    # Setting up the environment
    rltorch.set_seed(config['seed'])
    print("Setting up environment...", end=" ")
    env = E.TorchWrap(gym.make(config['environment_name']))
    env.seed(config['seed'])
    print("Done.")
      
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Logging
    logwriter = rltorch.log.LogWriter(SummaryWriter())

    # Setting up the networks
    device = torch.device("cuda:0" if torch.cuda.is_available() and not config['disable_cuda'] else "cpu")
    policy_net = rn.Network(Policy(state_size, action_size), 
                      torch.optim.Adam, config, device=device, name="Policy")
    value_net = rn.Network(Value(state_size), 
                      torch.optim.Adam, config, device=device, name="DQN")


    # Memory stores experiences for later training
    memory = M.EpisodeMemory()

    # Actor takes a net and uses it to produce actions from given states
    actor = StochasticSelector(policy_net, action_size, memory, device=device)

    # Agent is what performs the training
    agent = rltorch.agents.PPOAgent(policy_net, value_net, memory, config)

    # Runner performs a certain number of steps in the environment
    runner = rltorch.env.EnvironmentEpisodeSync(env, actor, config, name="Training", memory=memory, logwriter=logwriter)
    
    print("Training...")
    train(runner, agent, config, logwriter=logwriter) 

  # For profiling...
  # import cProfile
  # cProfile.run('train(runner, agent, config, logwriter = logwriter )')
  # python -m torch.utils.bottleneck /path/to/source/script.py [args] is also a good solution...

    print("Training Finished.")

    print("Evaluating...")
    rltorch.env.simulateEnvEps(env, actor, config, total_episodes=config['total_evaluation_episodes'], name="Evaluation")
    print("Evaulations Done.")

    logwriter.close() # We don't need to write anything out to disk anymore
