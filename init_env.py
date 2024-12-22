import numpy as np
from gym.spaces import Box
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym import ObservationWrapper,Wrapper
from gym.wrappers import TransformObservation, FrameStack, GrayScaleObservation
from skimage import transform


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


def init_env(level):
    level_parts = level.split('-')
    if len(level_parts) == 2:
        world = level_parts[0]
        stage = level_parts[1]
        env_name = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(env_name, render_mode="rgb_array", apply_api_compatibility=True)
    env = JoypadSpace(
        env,
        [['right'],
         ['right', 'A']]
    )
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    return env
