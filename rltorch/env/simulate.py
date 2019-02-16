from copy import deepcopy 
import rltorch

def simulateEnvEps(env, actor, config, total_episodes = 1, memory = None, logger = None, name = ""):
  for episode in range(total_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
      action = actor.act(state)
      next_state, reward, done, _ = env.step(action)

      episode_reward = episode_reward + reward
      if memory is not None:
        memory.append(state, action, reward, next_state, done)
      state = next_state

    if episode % config['print_stat_n_eps'] == 0:
      print("episode: {}/{}, score: {}"
        .format(episode, total_episodes, episode_reward))
    
    if logger is not None:
      logger.append(name + '/EpisodeReward', episode_reward)


class EnvironmentRunSync():
  def __init__(self, env, actor, config, memory = None, logwriter = None, name = ""):
    self.env = env
    self.name = name
    self.actor = actor
    self.config = deepcopy(config)
    self.logwriter = logwriter
    self.memory = memory
    self.episode_num = 1
    self.episode_reward = 0
    self.last_state = env.reset()

  def run(self, iterations):
    state = self.last_state
    logger = rltorch.log.Logger() if self.logwriter is not None else None
    for _ in range(iterations):
      action = self.actor.act(state)
      next_state, reward, done, _ = self.env.step(action)
       
      self.episode_reward += reward
      if self.memory is not None:
        self.memory.append(state, action, reward, next_state, done)
       
      state = next_state

      if done:
        if self.episode_num % self.config['print_stat_n_eps'] == 0:
          print("episode: {}/{}, score: {}"
            .format(self.episode_num, self.config['total_training_episodes'], self.episode_reward))
          
        if self.logwriter is not None:
          logger.append(self.name + '/EpisodeReward', self.episode_reward)
        self.episode_reward = 0
        state = self.env.reset()
        self.episode_num +=  1
          
    if self.logwriter is not None:
      self.logwriter.write(logger)
    
    self.last_state = state