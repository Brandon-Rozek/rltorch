from copy import deepcopy
import torch.multiprocessing as mp

class EnvironmentRun(mp.Process):
  def __init__(self, env, actor, config, memory = None, logger = None, name = ""):
    super(EnvironmentRun, self).__init__()
    self.env = env
    self.actor = actor
    self.memory = memory
    self.config = deepcopy(config)
    self.logger = logger
    self.name = name
    self.episode_num = 1
    self.episode_reward = 0
    self.last_state = env.reset()

  def run(self, iterations = 1, printstat = False):
    state = self.last_state
    for _ in range(iterations):
      action = self.actor.act(state)
      next_state, reward, done, _ = self.env.step(action)

      self.episode_reward = self.episode_reward + reward
      if self.memory is not None:
        self.memory.append(state, action, reward, next_state, done)
      state = next_state

      if done:
        if printstat:
            print("episode: {}/{}, score: {}"
                .format(self.episode_num, self.config['total_training_episodes'], self.episode_reward))
        if self.logger is not None:
          self.logger.append(self.name + '/EpisodeReward', self.episode_reward)
        self.episode_num = self.episode_num + 1
        self.episode_reward = 0
        state = self.env.reset()
  
    self.last_state = state
    
