import math


class Config:
    """
    配置类，用于存储训练过程中使用的超参数以及更新相关信息。
    """

    def __init__(self, buffer_size, batch_size):
        # 初始化配置参数
        self.best_reward = -float('inf')  # 记录训练过程中的最佳奖励，初始为负无穷
        self.buffer_size = buffer_size  # 经验回放缓冲区的大小
        self.batch_size = batch_size  # 批量大小，决定每次从回放缓冲区中采样的经验数量
        self.learning_times = 10000  # 每次训练过程中的最大学习次数
        self.training_episodes = 0  # 当前训练的回合数（episode计数器）
        self.num_episodes = 50000  # 总训练回合数
        self.model_counter = 0  # 用于存储模型保存时的计数器
        self.epsilon = 0.99  # epsilon值，用于epsilon-greedy策略，初始时值较高，表示更多探索
        self.beta = 0.5  # beta值，通常用于经验回放中的优先经验回放策略

    def training_episode(self):
        """
        返回当前训练的回合数。
        """
        return self.training_episodes

    def learning_times(self):
        """
        返回每次训练中最大学习次数（固定的10000）。
        """
        return self.learning_times

    def model_counter(self):
        """
        返回当前模型保存时的计数器并更新该计数器。
        """
        self.model_counter += 1
        return self.model_counter

    def num_episodes(self):
        """
        返回总的训练回合数。
        """
        return self.num_episodes

    def best_reward(self):
        """
        返回当前已记录的最佳奖励。
        """
        return self.best_reward

    def buffer_size(self):
        """
        返回经验回放缓冲区的大小。
        """
        return self.buffer_size

    def batch_size(self):
        """
        返回每次训练时使用的批量大小。
        """
        return self.batch_size

    def epsilon(self):
        """
        返回当前的epsilon值，用于控制探索和利用的平衡。
        """
        return self.epsilon

    def beta(self):
        """
        返回当前的beta值，用于优先经验回放中的权重更新。
        """
        return self.beta

    def update(self, replay_buffer):
        """
        更新训练回合数和 epsilon、beta 等超参数。
        - 增加训练回合数。
        - 根据回放缓冲区的大小调整 beta 和 epsilon。
        :param replay_buffer: 经验回放缓冲区对象
        """
        self.training_episodes += 1  # 增加训练回合数
        # 如果回放缓冲区中有足够的经验，则更新 beta
        if len(replay_buffer) > self.batch_size:
            # 根据训练回合数线性增加beta的值，beta最大为1.0
            self.beta = min(1.0, 0.5 + self.training_episodes * 6e-5)
        else:
            # 否则，将beta保持为0.5
            self.beta = 0.5

        # 使用指数衰减的方式更新epsilon，逐渐减小探索的比例
        self.epsilon = 0.1 + 0.95 * math.exp(-1.0 * ((self.training_episodes + 1) / 100000))
