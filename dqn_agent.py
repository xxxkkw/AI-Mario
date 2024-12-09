import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from random import random, randrange
from replay_buffer import ReplayBuffer
from torch import FloatTensor, LongTensor



class DQNAgent:
    def __init__(self, config, input, action_size, model_path=None):
        self.config = config
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
        self.training_model = self.build_net(input, action_size).to(self.device)
        self.target_model = self.build_net(input, action_size).to(self.device)
        # 优化器
        self.optimizer = optim.Adam(self.training_model.parameters(), lr=self.learning_rate)
        # 经验回放缓冲区
        self.memory = ReplayBuffer(self.buffer_size)  # 存储状态转移的经验，最大长度为 buffer_size
        self.update_counter = 0  # 更新计数器
        # 用于存储上一次的分数
        self.previous_score = 0
        if model_path:
            self.loda_model(model_path)

    def build_net(self, input, action_size):
        return nn.Sequential(
            nn.Conv2d(input[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        ).to(self.device)

    def forward(self, input):
        if len(input.shape) == 3:
            input = input.unsqueeze(0)
        return self.training_model(input)

    def replay(self, config, model, target_model, replay_buffer):
        """从回放记忆中采样并训练网络"""
        if len(replay_buffer) > self.learning_times:
            if not config.training_episodes % self.learning_times:
                target_model.load_state_dict(model.state_dict())
            self.optimizer.zero_grad()

            sample = replay_buffer.sample(self.batch_size, config.beta)
            batch, indices, weights = sample
            states = np.concatenate(batch[0])
            actions = batch[1]
            rewards = batch[2]
            next_states = np.concatenate(batch[3])
            dones = batch[4]

            state = FloatTensor(np.float32(states)).unsqueeze(1).to(self.device)
            next_state = FloatTensor(np.float32(next_states)).unsqueeze(1).to(self.device)
            action = LongTensor(actions).to(self.device)
            reward = FloatTensor(rewards).to(self.device)
            done = FloatTensor(dones).to(self.device)
            weights = FloatTensor(weights).to(self.device)
            q_values = model(state)
            next_q_values = target_model(next_state)

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
            state = state.unsqueeze(0).unsqueeze(0).float().to(self.device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        return action

    def agent_act(self, state):
        state = state.unsqueeze(0).unsqueeze(0).float().to(self.device)
        q_value = self.forward(state)
        action = q_value.max(1)[1].item()
        return action

    def loda_model(self, model_path):
        self.training_model.load_state_dict(torch.load(model_path))
        self.training_model.eval()

    def ai_act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        q_values = self.forward(state_tensor)
        action = q_values.max(1)[1].item()
        return action


