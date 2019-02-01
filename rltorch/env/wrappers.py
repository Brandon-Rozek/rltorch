import gym
import torch
from gym import spaces
import cv2
from collections import deque

# Mostly derived from OpenAI baselines
class FireResetEnv(gym.Wrapper):
  def __init__(self, env):
    """Take action on reset for environments that are fixed until firing."""
    gym.Wrapper.__init__(self, env)
    assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    assert len(env.unwrapped.get_action_meanings()) >= 3

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(1)
    if done:
      self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(2)
    if done:
      self.env.reset(**kwargs)
    return obs

  def step(self, ac):
    return self.env.step(ac)

class LazyFrames(object):
  def __init__(self, frames):
    """This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was."""
    self._frames = frames
    self._out = None

  def _force(self):
    if self._out is None:
        self._out = torch.stack(self._frames)
        self._frames = None
    return self._out
 
  def __array__(self, dtype=None):
    out = self._force()
    if dtype is not None:
        out = out.astype(dtype)
    return out

  def __len__(self):
    return len(self._force())

  def __getitem__(self, i):
    return self._force()[i]

class FrameStack(gym.Wrapper):
  def __init__(self, env, k):
    """Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    # return LazyFrames(list(self.frames))
    return torch.cat(list(self.frames)).unsqueeze(0)

class ProcessFrame(gym.Wrapper):
  def __init__(self, env, resize_shape = None, crop_bounds = None, grayscale = False):
    gym.Wrapper.__init__(self, env)
    self.resize_shape = resize_shape
    self.crop_bounds = crop_bounds
    self.grayscale = grayscale
    
  def reset(self):
    return self._preprocess(self.env.reset())
  
  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    next_state = self._preprocess(next_state)
    return next_state, reward, done, info
  
  def _preprocess(self, frame):
    if self.grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if self.crop_bounds is not None and len(self.crop_bounds) == 4:
        frame = frame[self.crop_bounds[0]:self.crop_bounds[1], self.crop_bounds[2]:self.crop_bounds[3]] 
    if self.resize_shape is not None and len(self.resize_shape) == 2:
        frame = cv2.resize(frame, self.resize_shape, interpolation=cv2.INTER_AREA)
    # Normalize
    frame = frame / 255
    return frame


# Turns observations into torch tensors
# Adds an additional dimension that's suppose to represent the batch dim
class TorchWrap(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)

  def reset(self):
    return self._convert(self.env.reset())
  
  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    next_state = self._convert(next_state)
    return next_state, reward, done, info

  def _convert(self, frame):
    frame = torch.from_numpy(frame).unsqueeze(0).float()
    return frame