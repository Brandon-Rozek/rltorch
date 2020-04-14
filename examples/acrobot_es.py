import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
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
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 125)
        self.fc_norm = nn.LayerNorm(125)

        self.fc2 = nn.Linear(125, 125)
        self.fc2_norm = nn.LayerNorm(125)
        self.action_prob = nn.Linear(125, action_size)

    def forward(self, x):
        x = F.relu(self.fc_norm(self.fc1(x)))
        x = F.relu(self.fc2_norm(self.fc2(x)))
        x = F.softmax(self.action_prob(x), dim = 1)
        return x

#
## Configuration
#
config = {}
config['seed'] = 901
config['environment_name'] = 'Acrobot-v1'
config['total_training_episodes'] = 50
config['total_evaluation_episodes'] = 5
config['learning_rate'] = 1e-1
config['discount_rate'] = 0.99
# How many episodes between printing out the episode stats
config['print_stat_n_eps'] = 1
config['disable_cuda'] = False

#
## Training Loop
#
def train(runner, net, config, logwriter=None):
    finished = False
    while not finished:
        runner.run()
        net.calc_gradients()
        net.step()
        if logwriter is not None:
            net.log_named_parameters()
            logwriter.write(Logger)
        finished = runner.episode_num > config['total_training_episodes']
 
#
## Loss function
#
def fitness(model):
    env = gym.make("Acrobot-v1")
    state = torch.from_numpy(env.reset()).float().unsqueeze(0)
    total_reward = 0
    done = False
    while not done:
        action_probabilities = model(state)
        distribution = Categorical(action_probabilities)
        action = distribution.sample().item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = torch.from_numpy(next_state).float().unsqueeze(0)
    return -total_reward

if __name__ == "__main__":
    # Hide internal gym warnings
    gym.logger.set_level(40)

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
    net = rn.ESNetwork(Policy(state_size, action_size),
                        torch.optim.Adam, 100, fitness, config, device=device, name="ES")  
    # Actor takes a net and uses it to produce actions from given states
    actor = StochasticSelector(net, action_size, device=device)   
    # Runner performs an episode of the environment
    runner = rltorch.env.EnvironmentEpisodeSync(env, actor, config, name="Training", logwriter=logwriter)   
    print("Training...")
    train(runner, net, config, logwriter=logwriter)  
    # For profiling...
    # import cProfile
    # cProfile.run('train(runner, agent, config, logwriter = logwriter )')
    # python -m torch.utils.bottleneck /path/to/source/script.py [args] is also a good solution...
    print("Training Finished.") 
    print("Evaluating...")
    rltorch.env.simulateEnvEps(env, actor, config, total_episodes=config['total_evaluation_episodes'], name="Evaluation")
    print("Evaulations Done.")

    logwriter.close() # We don't need to write anything out to disk anymore
