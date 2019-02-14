# EnvironmentEpisode is currently under maintenance
# Feel free to use the old API, though it is scheduled to change soon.

from copy import deepcopy
import torch.multiprocessing as mp

class EnvironmentEpisode(mp.Process):
  def __init__(self, env, actor, config, logger = None, name = ""):
    super(EnvironmentEpisode, self).__init__()
    self.env = env
    self.actor = actor
    self.config = deepcopy(config)
    self.logger = logger
    self.name = name
    self.episode_num = 1

  def run(self, printstat = False, memory = None):
    state = self.env.reset()
    done = False
    episode_reward = 0
    while not done:
      action = self.actor.act(state)
      next_state, reward, done, _ = self.env.step(action)

      episode_reward = episode_reward + reward
      if memory is not None:
        memory.put((state, action, reward, next_state, done))
      state = next_state

    if printstat:
      print("episode: {}/{}, score: {}"
        .format(self.episode_num, self.config['total_training_episodes'], episode_reward))
    if self.logger is not None:
      self.logger.append(self.name + '/EpisodeReward', episode_reward)

    self.episode_num += 1








# from copy import deepcopy
# import torch.multiprocessing as mp
# from ctypes import *
# import rltorch.log

# def envepisode(actor, env, episode_num, config, runcondition, memoryqueue = None, logqueue = None, name = ""):
#   # Wait for signal to start running through the environment
#   while runcondition.wait():
#     # Start a logger to log the rewards
#     logger = rltorch.log.Logger()
#     state = env.reset()
#     episode_reward = 0
#     done = False
#     while not done:
#       action = actor.act(state)
#       next_state, reward, done, _ = env.step(action)
       
#       episode_reward += reward
#       if memoryqueue is not None:
#         memoryqueue.put((state, action, reward, next_state, done))
       
#       state = next_state

#       if done:
#         with episode_num.get_lock():
#           if episode_num.value % config['print_stat_n_eps'] == 0:
#             print("episode: {}/{}, score: {}"
#               .format(episode_num.value, config['total_training_episodes'], episode_reward))
          
#         if logger is not None:
#           logger.append(name + '/EpisodeReward', episode_reward)
#         episode_reward = 0
#         state = env.reset()
#         with episode_num.get_lock():
#           episode_num.value +=  1
          
#     logqueue.put(logger)
  
# class EnvironmentRun():
#   def __init__(self, env_func, actor, config, memory = None, name = ""):
#     self.config = deepcopy(config)
#     self.memory = memory
#     self.episode_num = mp.Value(c_uint)
#     self.runcondition = mp.Event()
#     # Interestingly enough, there isn't a good reliable way to know how many states an episode will have
#     # Perhaps we can share a uint to keep track...
#     self.memory_queue = mp.Queue(maxsize = config['replay_skip'] + 1)
#     self.logqueue = mp.Queue(maxsize = 1)
#     with self.episode_num.get_lock():
#       self.episode_num.value = 1
#     self.runner = mp.Process(target=envrun, 
#       args=(actor, env_func, self.episode_num, config, self.runcondition),
#       kwargs = {'iterations': config['replay_skip'] + 1, 
#         'memoryqueue' : self.memory_queue, 'logqueue' : self.logqueue, 'name' : name})
#     self.runner.start()

#   def run(self):
#     self.runcondition.set()

#   def join(self):
#     self._sync_memory()
#     if self.logwriter is not None:
#       self.logwriter.write(self._get_reward_logger())

#   def sync_memory(self):
#     if self.memory is not None:
#       for i in range(self.config['replay_skip'] + 1):
#         self.memory.append(*self.memory_queue.get())

#   def get_reward_logger(self):
#     return self.logqueue.get()

#   def terminate(self):
#     self.runner.terminate()
    
