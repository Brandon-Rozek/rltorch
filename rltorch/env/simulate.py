from copy import deepcopy
import time
import rltorch

def simulateEnvEps(env, actor, config, total_episodes=1, memory=None, name="", render=False):
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = actor.act(state)
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()
                time.sleep(0.01)

        episode_reward = episode_reward + reward
        if memory is not None:
            memory.append(state, action, reward, next_state, done)
        state = next_state

        if episode % config['print_stat_n_eps'] == 0:
            print("episode: {}/{}, score: {}"
                  .format(episode, total_episodes, episode_reward), flush=True)
    
        if rltorch.log.enabled:
            rltorch.log.Logger[name + '/EpisodeReward'].append(episode_reward)


class EnvironmentRunSync:
    def __init__(self, env, actor, config, memory=None, logwriter=None, name="", render=False):
        self.env = env
        self.name = name
        self.actor = actor
        self.config = deepcopy(config)
        self.logwriter = logwriter
        self.memory = memory
        self.episode_num = 1
        self.episode_reward = 0
        self.last_state = env.reset()
        self.render = render

    def run(self, iterations):
        state = self.last_state
        for _ in range(iterations):
            action = self.actor.act(state)
            next_state, reward, done, _ = self.env.step(action)
            if self.render:
                self.env.render()
       
            self.episode_reward += reward
            if self.memory is not None:
                self.memory.append(state, action, reward, next_state, done)
       
            state = next_state

            if done:
                if self.episode_num % self.config['print_stat_n_eps'] == 0:
                    print("episode: {}/{}, score: {}"
                          .format(self.episode_num, self.config['total_training_episodes'], self.episode_reward), flush=True)
          
                if self.logwriter is not None:
                    rltorch.log.Logger[self.name + '/EpisodeReward'].append(self.episode_reward)
                self.episode_reward = 0
                state = self.env.reset()
                self.episode_num += 1
          
                if self.logwriter is not None:
                    self.logwriter.write(rltorch.log.Logger)
    
        self.last_state = state


class EnvironmentEpisodeSync:
    def __init__(self, env, actor, config, memory=None, logwriter=None, name=""):
        self.env = env
        self.name = name
        self.actor = actor
        self.config = deepcopy(config)
        self.logwriter = logwriter
        self.memory = memory
        self.episode_num = 1

    def run(self):
        state = self.env.reset()
        done = False
        episodeReward = 0
        while not done:
            action = self.actor.act(state)
            next_state, reward, done, _ = self.env.step(action)
            episodeReward += reward
            if self.memory is not None:
                self.memory.append(state, action, reward, next_state, done)

            state = next_state

        if self.episode_num % self.config['print_stat_n_eps'] == 0:
            print("episode: {}/{}, score: {}"
                  .format(self.episode_num, self.config['total_training_episodes'], episodeReward), flush=True)
          
        if self.logwriter is not None:
            rltorch.log.Logger[self.name + '/EpisodeReward'].append(episodeReward)
            self.logwriter.write(rltorch.log.Logger)
    
        self.episode_num += 1
