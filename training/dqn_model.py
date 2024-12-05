import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pygame
import time

import config
from data.states import level1
from collections import deque

keybinding = {
    'action': pygame.K_s,
    'jump': pygame.K_a,
    'left': pygame.K_LEFT,
    'right': pygame.K_RIGHT,
    'down': pygame.K_DOWN
}

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, action_size, state_size=(84, 84, 1), gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=0.00025, buffer_size=10000, batch_size=64,
                 target_update_freq=10):
        self.action_size = action_size  # 动作空间大小
        self.state_size = state_size  # 图像输入的尺寸
        self.gamma = gamma  # 折扣因子，决定长期奖励的权重
        self.epsilon = epsilon  # epsilon-greedy 策略中的 epsilon，控制探索和利用
        self.epsilon_min = epsilon_min  # epsilon 的最小值
        self.epsilon_decay = epsilon_decay  # epsilon 的衰减率
        self.learning_rate = learning_rate  # 学习率
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # 创建 Q 网络和目标网络，并将它们转移到设备上
        self.model = self.build_model().to(self.device)  # 主 Q 网络
        self.target_model = self.build_model().to(self.device)  # 目标网络（用于更新目标 Q 值）
        self.update_target_model()  # 初始化时同步目标网络

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(buffer_size)  # 存储状态转移的经验，最大长度为 buffer_size
        self.batch_size = batch_size  # 批处理大小

        # 控制目标网络更新频率
        self.target_update_freq = target_update_freq
        self.update_counter = 0  # 更新计数器
        # 用于存储上一次的分数
        self.previous_score = 0

    def build_model(self):
        """构建卷积神经网络模型"""
        model = nn.Sequential(
            nn.Conv2d(self.state_size[2], 32, kernel_size=8, stride=4),  # 第一个卷积层
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 第二个卷积层
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 第三个卷积层
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),  # 全连接层
            nn.ReLU(),
            nn.Linear(512, self.action_size)  # 输出层，输出 Q 值
        )
        return model

    def update_target_model(self):
        """将 Q 网络的权重复制到目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def store_experience(self, state, action, reward, next_state, done):
        """
        存储一条经验到经验回放缓冲区。
        """
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            actions = [0, 1, 2, 3, 4]
            # action jump left right down
            probabilities = [0, 0.5, 0, 0.5, 0]
            chosen_action = np.random.choice(actions, p=probabilities)
            config.random_count += 1
        else:
            # 利用模型选择动作
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            state = state.permute(0, 3, 1, 2)
            q_values = self.model(state)
            chosen_action = torch.argmax(q_values[0]).item()
            config.model_count += 1
        return chosen_action

    def replay(self):
        """从回放记忆中采样并训练网络"""
        if len(self.memory.buffer) < self.batch_size:
            return

        # 从记忆中随机采样
        minibatch = self.memory.sample(self.batch_size)

        # 强制将所有的张量移到同一设备上
        states_numpy = np.array([x[0] for x in minibatch])
        next_states_numpy = np.array([x[3] for x in minibatch])
        states = torch.FloatTensor(states_numpy).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)  # 动作的原始格式
        rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
        next_states = torch.FloatTensor(next_states_numpy).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor([x[4] for x in minibatch]).to(self.device)



        # 计算目标 Q 值
        target_q_values = self.target_model(next_states)
        max_target_q_values = torch.max(target_q_values, dim=1)[0]  # 每个样本的最大 Q 值
        targets = rewards + self.gamma * max_target_q_values * (1 - dones)  # 每个状态下的目标Q值 由 Bellman 方程解得
        current_q_values = self.model(states)
        actions = actions.reshape(-1, 1)
        current_q_values = current_q_values.gather(1, actions)
        targets = targets.reshape(-1, 1)
        # 计算损失
        loss = nn.MSELoss()(current_q_values, targets)
        # 反向传播更新模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """加载已保存的模型权重"""
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        """保存模型权重"""
        torch.save(self.model.state_dict(), name)
