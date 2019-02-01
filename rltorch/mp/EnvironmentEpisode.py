from copy import deepcopy
import torch.multiprocessing as mp

class EnvironmentEpisode(mp.Process):
  def __init__(self, env, actor, config, memory = None, logger = None, name = ""):
    super(EnvironmentEpisode, self).__init__()
    self.env = env
    self.actor = actor
    self.memory = memory
    self.config = deepcopy(config)
    self.logger = logger
    self.name = name
    self.episode_num = 1

  def run(self, printstat = False):
    state = self.env.reset()
    done = False
    episode_reward = 0
    while not done:
      action = self.actor.act(state)
      next_state, reward, done, _ = self.env.step(action)

      episode_reward = episode_reward + reward
      if self.memory is not None:
        self.memory.append(state, action, reward, next_state, done)
      state = next_state

    if printstat:
      print("episode: {}/{}, score: {}"
        .format(self.episode_num, self.config['total_training_episodes'], episode_reward))
    if self.logger is not None:
      self.logger.append(self.name + '/EpisodeReward', episode_reward)

    self.episode_num += 1

