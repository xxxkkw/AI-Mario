import torch
import numpy as np
from gym.spaces import Box
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym import ObservationWrapper, Wrapper
import torchvision.transforms as T


class SkipFrame(Wrapper):
    """
    环境的封装器，允许跳过一定数量的帧以加快训练速度。
    """

    def __init__(self, env, skip):
        """
        :param env: 环境实例
        :param skip: 每次调用step时跳过的帧数
        """
        super().__init__(env)
        self._skip = skip  # 设置每次跳过的帧数

    def step(self, action):
        """
        执行动作并跳过一定数量的帧。
        :param action: 环境中执行的动作
        :return: 观察、总奖励、是否结束、信息等
        """
        total_reward = 0.0
        # 跳过多个帧
        for i in range(self._skip):
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(ObservationWrapper):
    """
    环境的封装器，将图像观察转换为灰度图像。
    """

    def __init__(self, env):
        """
        :param env: 环境实例
        """
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]  # 获取原始观察图像的宽度和高度
        # 设置新的观察空间为灰度图像（只有两个维度：宽度和高度）
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        """
        将图像的通道维度从最后一个维度移到第一个维度，并转换为PyTorch张量。
        :param observation: 输入的图像（numpy数组）
        :return: 处理后的图像（PyTorch张量）
        """
        observation = np.transpose(observation, (2, 0, 1))  # 转置图像维度
        observation = torch.tensor(observation.copy(), dtype=torch.float)  # 转换为PyTorch张量
        return observation

    def observation(self, observation):
        """
        将观察图像转换为灰度图。
        :param observation: 输入的彩色图像
        :return: 转换后的灰度图像
        """
        observation = self.permute_orientation(observation)  # 转换图像维度
        transform = T.Grayscale()  # 定义灰度转换
        observation = transform(observation)  # 应用灰度转换
        return observation


class ResizeObservation(ObservationWrapper):
    """
    环境的封装器，用于将观察图像调整为指定的大小。
    """

    def __init__(self, env, shape):
        """
        :param env: 环境实例
        :param shape: 目标图像的大小，单个整数表示方形图像，或二元组表示宽和高
        """
        super().__init__(env)
        # 如果形状是单个整数，则表示方形图像
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # 更新观察空间的形状，以适应新的图像大小
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """
        将观察图像调整为指定的大小，并进行归一化。
        :param observation: 输入的观察图像
        :return: 调整后的图像
        """
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]  # 使用Resize和Normalize进行转换
        )
        observation = transforms(observation).squeeze(0)  # 调整大小并去掉额外的维度
        return observation


def init_env(type):
    """
    初始化并返回Super Mario Bros环境。
    :param type: 环境类型，'running'为可视化环境，'training'为不渲染图像环境
    :return: 初始化后的环境实例
    """
    # 根据环境类型设置渲染模式
    if type == "running":
        # 创建一个带渲染的环境
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True)
    else:
        env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb_array", apply_api_compatibility=True)

    # 使用JoypadSpace包装环境，限制动作集为SIMPLE_MOVEMENT
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # 添加跳帧功能，跳过4帧
    env = SkipFrame(env, skip=4)

    # 将环境的观察转换为灰度图
    env = GrayScaleObservation(env)

    # 将观察图像调整为84x84大小
    env = ResizeObservation(env, shape=84)

    return env
