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

