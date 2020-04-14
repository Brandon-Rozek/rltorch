from copy import deepcopy
from ctypes import c_uint
import torch.multiprocessing as mp
import rltorch.log

def envrun(actor, env, episode_num, config, runcondition, iterations=1, memoryqueue=None, logqueue=None, name=""):
    state = env.reset()
    episode_reward = 0
    # Wait for signal to start running through the environment
    while runcondition.wait():
      # Start a logger to log the rewards
        logger = rltorch.log.Logger() if logqueue is not None else None
    for _ in range(iterations):
        action = actor.act(state)
        next_state, reward, done, _ = env.step(action)
     
        episode_reward += reward
        if memoryqueue is not None:
            memoryqueue.put((state, action, reward, next_state, done))
     
        state = next_state
    
        if done:
            with episode_num.get_lock():
                if episode_num.value % config['print_stat_n_eps'] == 0:
                    print("episode: {}/{}, score: {}"
                        .format(episode_num.value, config['total_training_episodes'], episode_reward))
        
            if logger is not None:
                logger.append(name + '/EpisodeReward', episode_reward)
            episode_reward = 0
            state = env.reset()
            with episode_num.get_lock():
                episode_num.value +=  1
        
            if logqueue is not None:
                logqueue.put(logger)
  
class EnvironmentRun():
  def __init__(self, env, actor, config, memory = None, logwriter = None, name = ""):
    self.config = deepcopy(config)
    self.logwriter = logwriter
    self.memory = memory
    self.episode_num = mp.Value(c_uint)
    self.runcondition = mp.Event()
    self.memory_queue = mp.Queue(maxsize = config['replay_skip'] + 1) if memory is not None else None
    self.logqueue = mp.Queue(maxsize = 1) if logwriter is not None else None
    with self.episode_num.get_lock():
      self.episode_num.value = 1
    self.runner = mp.Process(target=envrun, 
      args=(actor, env, self.episode_num, config, self.runcondition),
      kwargs = {'iterations': config['replay_skip'] + 1, 
        'memoryqueue' : self.memory_queue, 'logqueue' : self.logqueue, 'name' : name})
    self.runner.start()

  def run(self):
    self.runcondition.set()
  
  def join(self):
    self._sync_memory()
    if self.logwriter is not None:
      self.logwriter.write(self._get_reward_logger())

  def _sync_memory(self):
    if self.memory is not None:
      for i in range(self.config['replay_skip'] + 1):
        self.memory.append(*self.memory_queue.get())

  def _get_reward_logger(self):
    if self.logqueue is not None:
      return self.logqueue.get()

  def terminate(self):
    self.runner.terminate()
    
