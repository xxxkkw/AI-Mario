import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from random import random, randrange
from replay_buffer import ReplayBuffer
from torch import FloatTensor, LongTensor


class DQNAgent(nn.Module):
    def __init__(self, config, input, action_size, model_path=None):
        super().__init__()
        self.input = input
        self.action_size = action_size
        self.gamma = 0.99  # 折扣因子，决定长期奖励的权重
        self.epsilon = 1.0  # epsilon-greedy 策略中的 epsilon，控制探索和利用
        self.epsilon_min = 0.1  # epsilon 的最小值
        self.epsilon_decay = 0.99999975  # epsilon 的衰减率
        self.learning_rate = 0.00025  # 学习率
        self.buffer_size = 20000
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.learning_times = 10000  # 更新模型的尝试次数
        self.episodes = 50000
        self.update_frequency = 1000
        self.batch_size = 128
        self.target_update_freq = 10
        # 优化器

        self.online = nn.Sequential(
            nn.Conv2d(input[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.learning_rate)
        # 经验回放缓冲区
        self.memory = ReplayBuffer(self.buffer_size)  # 存储状态转移的经验，最大长度为 buffer_size
        self.update_counter = 0  # 更新计数器
        # 用于存储上一次的分数
        self.previous_score = 0

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

        if model_path:
            state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
            self.online.load_state_dict(state_dict)

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)

    def replay(self, config, online, target, replay_buffer):
        """从回放记忆中采样并训练网络"""
        if len(replay_buffer) > self.learning_times:
            if not config.training_episodes % self.learning_times:
                target.load_state_dict(online.state_dict())
            self.optimizer.zero_grad()

            sample = replay_buffer.sample(self.batch_size, config.beta)
            batch, indices, weights = sample

            actions = batch[1]
            rewards = batch[2]
            dones = batch[4]

            states = FloatTensor(np.float32(batch[0])).to(self.device)
            next_states = FloatTensor(np.float32(batch[3])).to(self.device)
            action = LongTensor(actions).to(self.device)
            reward = FloatTensor(rewards).to(self.device)
            done = FloatTensor(dones).to(self.device)
            weights = FloatTensor(weights).to(self.device)
            q_values = online(states)
            next_q_values = target(next_states)

            q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + self.gamma * next_q_value * (1 - done)
            loss = ((q_value - target_q_value.detach()).pow(2) * weights).mean()

            if np.isnan(loss.item()) or np.isinf(loss.item()):
                loss = torch.tensor(1.0, device=self.device)

            loss.backward()
            self.optimizer.step()

            loss = loss.detach().cpu().numpy()
            loss = np.repeat(loss, len(indices))
            replay_buffer.update_priorities(indices, loss + 1e5)

    def act(self, state, epsilon):
        if random() < epsilon:
            action = randrange(self.action_size)
        else:
            state = torch.FloatTensor(state).to(self.device)
            state = state.unsqueeze(0)
            q_values = self.online(state)
            action = torch.argmax(q_values, axis=1).item()
        return action


    def agent_act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        state = state.unsqueeze(0)
        q_values = self.online(state)
        action = torch.argmax(q_values, axis=1).item()
        return action