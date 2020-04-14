# EnvironmentEpisode is currently under maintenance
# Feel free to use the old API, though it is scheduled to change soon.

from copy import deepcopy
import torch.multiprocessing as mp
import rltorch.log as log

class EnvironmentEpisode(mp.Process):
    def __init__(self, env, actor, config, name=""):
        super(EnvironmentEpisode, self).__init__()
        self.env = env
        self.actor = actor
        self.config = deepcopy(config)
        self.name = name
        self.episode_num = 1

    def run(self, printstat=False, memory=None):
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
        if log.enabled:
            log.Logger[self.name + '/EpisodeReward'].append(episode_reward)

        self.episode_num += 1
