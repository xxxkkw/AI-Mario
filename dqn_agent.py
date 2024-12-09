import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from random import random, randrange
from replay_buffer import ReplayBuffer
from torch import FloatTensor, LongTensor


class DQNAgent:
    def __init__(self, config, input, action_size, model_path=None):
        """
        初始化 DQN 智能体
        :param config: 配置参数（包括训练的相关设置）
        :param input: 输入的状态维度
        :param action_size: 动作空间大小
        :param model_path: 如果给定路径，则加载已有的模型
        """
        self.config = config
        self.input = input
        self.action_size = action_size
        self.gamma = 0.99  # 折扣因子，决定长期奖励的权重
        self.epsilon = 1.0  # epsilon-greedy 策略中的 epsilon，控制探索和利用
        self.epsilon_min = 0.1  # epsilon 的最小值
        self.learning_rate = 0.00025  # 学习率
        self.buffer_size = 20000  # 经验回放缓冲区的大小
        # 选择设备：如果有 CUDA 支持则用 GPU，否则使用 MPS 或 CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.learning_times = 10000  # 更新模型的尝试次数
        self.episodes = 50000  # 最大训练回合数
        self.update_frequency = 1000  # 每隔多少回合更新一次
        self.batch_size = 128  # 每次训练的批量大小
        self.target_update_freq = 10  # 更新目标模型的频率
        # 构建两个神经网络：训练模型和目标模型
        self.training_model = self.build_net(input, action_size).to(self.device)
        self.target_model = self.build_net(input, action_size).to(self.device)
        # 优化器：使用 Adam 算法
        self.optimizer = optim.Adam(self.training_model.parameters(), lr=self.learning_rate)
        # 经验回放缓冲区，用于存储状态转移数据
        self.memory = ReplayBuffer(self.buffer_size)
        self.update_counter = 0  # 更新计数器
        self.previous_score = 0  # 存储上一次的分数

        # 如果提供了模型路径，则加载预训练模型
        if model_path:
            self.load_model(model_path)

    def build_net(self, input, action_size):
        """
        构建 DQN 网络模型
        :param input: 输入状态维度
        :param action_size: 动作空间的大小
        :return: 构建好的神经网络
        """
        return nn.Sequential(
            # 输入卷积层，32个输出通道，大小为 8x8，步长为 4
            nn.Conv2d(input[0], 32, 8, 4),
            nn.ReLU(),
            # 第二个卷积层，64个输出通道，大小为 4x4，步长为 2
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            # 第三个卷积层，64个输出通道，大小为 3x3，步长为 1
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            # Flatten层，将多维输入展平成一维
            nn.Flatten(),
            # 全连接层，输入特征为 64 * 7 * 7，输出 512
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            # 输出层，大小为动作空间的大小
            nn.Linear(512, action_size)
        ).to(self.device)

    def forward(self, input):
        """
        网络的前向传播
        :param input: 输入状态
        :return: 输出Q值
        """
        # 如果输入是 3D 张量（单张图片），则增加一个 batch 维度
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
        # 进行前向传播
        return self.training_model(input)

    def replay(self, config, model, target_model, replay_buffer):
        """
        从回放记忆中采样并训练网络
        :param config: 配置信息
        :param model: 当前的训练模型
        :param target_model: 目标模型
        :param replay_buffer: 经验回放缓冲区
        """
        if len(replay_buffer) > self.learning_times:
            # 每经过一定次数的学习后，同步目标模型
            if not config.training_episodes % self.learning_times:
                target_model.load_state_dict(model.state_dict())
            self.optimizer.zero_grad()

            # 从回放缓冲区中随机采样
            sample = replay_buffer.sample(self.batch_size, config.beta)
            batch, indices, weights = sample
            states = np.concatenate(batch[0])
            actions = batch[1]
            rewards = batch[2]
            next_states = np.concatenate(batch[3])
            dones = batch[4]

            # 转换为 tensor，并移动到合适的设备
            state = FloatTensor(np.float32(states)).unsqueeze(1).to(self.device)
            next_state = FloatTensor(np.float32(next_states)).unsqueeze(1).to(self.device)
            action = LongTensor(actions).to(self.device)
            reward = FloatTensor(rewards).to(self.device)
            done = FloatTensor(dones).to(self.device)
            weights = FloatTensor(weights).to(self.device)

            # 计算当前Q值和目标Q值
            q_values = model(state)
            next_q_values = target_model(next_state)
            q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + self.gamma * next_q_value * (1 - done)

            # 计算损失
            loss = ((q_value - target_q_value.detach()).pow(2) * weights).mean()

            # 如果损失值为NaN或无穷大，设置一个默认的损失值
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                loss = torch.tensor(1.0, device=self.device)

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            # 更新回放缓冲区的优先级
            loss = loss.detach().cpu().numpy()
            loss = np.repeat(loss, len(indices))
            replay_buffer.update_priorities(indices, loss + 1e5)

    def act(self, state, epsilon):
        """
        根据当前状态选择动作，使用 epsilon-greedy 策略
        :param state: 当前状态
        :param epsilon: epsilon值
        :return: 选择的动作
        """
        if random() < epsilon:
            action = randrange(self.action_size)  # 随机选择动作
        else:
            state = state.unsqueeze(0).unsqueeze(0).float().to(self.device)  # 转换为适当格式
            q_value = self.forward(state)  # 计算当前状态的 Q 值
            action = q_value.max(1)[1].item()  # 选择具有最大Q值的动作
        return action

    def agent_act(self, state):
        """
        根据当前状态选择动作（用于推理或测试阶段）
        :param state: 当前状态
        :return: 选择的动作
        """
        state = state.unsqueeze(0).unsqueeze(0).float().to(self.device)
        q_value = self.forward(state)
        action = q_value.max(1)[1].item()
        return action

    def load_model(self, model_path):
        """
        加载预训练的模型
        :param model_path: 模型文件的路径
        """
        self.training_model.load_state_dict(torch.load(model_path))
        self.training_model.eval()  # 切换到评估模式
