import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import random
import numpy as np
from collections import deque
from training.dqn_model import DQN

keybinding = {
    'action': pygame.K_s,
    'jump': pygame.K_a,
    'left': pygame.K_LEFT,
    'right': pygame.K_RIGHT,
    'down': pygame.K_DOWN
}

class Agent:
    def __init__(self, action_size, state_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.learning_rate = 0.001  # 学习率

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # Q-Network
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        """
        更新目标网络
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        将经验存储到记忆中
        """
        self.memory.append((state, action, reward, next_state, done))

    # 假设你的按键映射如下：
    keybinding = {
        'action': pygame.K_s,
        'jump': pygame.K_a,
        'left': pygame.K_LEFT,
        'right': pygame.K_RIGHT,
        'down': pygame.K_DOWN
    }

    class Agent:
        def __init__(self, model, epsilon, device):
            self.model = model
            self.epsilon = epsilon
            self.device = device

        def act(self, state):
            """
            基于 epsilon-greedy 策略选择动作，并将动作映射到 pygame 按键
            """
            # 定义动作到按键的映射：0-向左，1-向右，2-跳跃，3-下蹲，4-特殊动作
            action_map = {
                0: keybinding['left'],  # 向左
                1: keybinding['right'],  # 向右
                2: keybinding['jump'],  # 跳跃
                3: keybinding['down'],  # 下蹲
                4: keybinding['action']  # 特殊动作/其他动作
            }

            if np.random.rand() <= self.epsilon:
                # 随机选择动作，带有概率权重（可以调整）
                actions = [0, 1, 2, 3, 4]  # 动作列表
                probabilities = [0.1, 0.5, 0.2, 0.1, 0.1]  # 权重，右走的概率最大
                chosen_action = np.random.choice(actions, p=probabilities)  # 按概率选择动作
            else:
                # 利用模型选择动作（Q-learning）
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                q_values = self.model(state)
                if q_values.dim() == 2:  # 输出形状是 [batch_size, action_size]
                    chosen_action = torch.argmax(q_values[0]).item()
                else:
                    chosen_action = torch.argmax(q_values).item()
            # 创建一个与 pygame.key.get_pressed() 返回值相同格式的元组 (长为 323)
            keys_state = [False] * 323  # 初始化所有按键为 False
            # 根据选择的动作设置对应的按键为 True
            if chosen_action in action_map:
                key_index = action_map[chosen_action]  # 获取对应的 pygame 按键码
                keys_state[key_index] = True  # 设置对应按键为 True（按下）
            # 通过 ScancodeWrapper 返回状态
            return pygame.key.ScancodeWrapper(*keys_state)

    def learn(self, state, action, reward, next_state, done):
        """
        训练并更新 Q 表
        """
        # 获取当前 Q 值
        current_q_values = self.model(state)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_model(next_state)
            max_next_q_value = next_q_values.max(1)[0]  # 获取下一个状态的最大 Q 值
            target_q_value = reward + (self.gamma * max_next_q_value * (1 - done))  # 更新的目标 Q 值

        # 更新 Q 网络
        current_q_values[0][action] = target_q_value  # 只更新选择的动作的 Q 值

        # 计算损失
        loss = nn.MSELoss()(self.model(state), current_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        """
        从记忆池中抽取一批经验进行训练
        """
        # 如果记忆池中经验不够一个批次，直接返回
        if len(self.memory) < batch_size:
            return

        # 随机抽取一批经验
        minibatch = random.sample(self.memory, batch_size)

        # 将每个元素从 minibatch 中分离出 (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 将数据转化为 torch 张量并移动到正确的设备
        states = torch.cat(states, dim=0).to(self.device)  # 批量处理状态
        next_states = torch.cat(next_states, dim=0).to(self.device)

        # 将 actions, rewards, dones 转换为张量
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        # 获取当前 Q 网络的输出
        current_q_values = self.model(states)

        # 获取目标 Q 网络的输出
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]  # 获取每个样本的最大 Q 值

        # 计算目标 Q 值
        target_q_values = current_q_values.clone()  # 克隆当前 Q 值，避免直接修改
        for i in range(batch_size):
            target_q_values[i][actions[i]] = rewards[i] + self.gamma * max_next_q_values[i] * (1 - dones[i])

        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



