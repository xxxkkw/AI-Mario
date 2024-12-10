import numpy as np


class ReplayBuffer:
    """
    经验回放缓冲区，存储智能体与环境交互过程中产生的状态转移，用于后续的训练。
    """

    def __init__(self, capacity):
        # 初始化回放缓冲区
        self.alpha = 0.6  # 优先经验回放中用于调整优先级的参数
        self.beta = 0.5  # 用于计算重要性采样权重的参数
        self.capacity = capacity  # 缓冲区的最大容量
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # 存储每个经验的优先级
        self.buffer = []  # 存储经验的列表
        self.position = 0  # 当前存储经验的位置

    def push(self, state, action, reward, next_state, done):
        """
        将一条经验（state, action, reward, next_state, done）推入经验回放缓冲区。
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 当前状态下采取该动作获得的奖励
        :param next_state: 执行动作后到达的下一状态
        :param done: 是否为回合结束的标志
        """
        # 检查state和next_state是否为tuple类型（通常是图像或其他结构）
        if isinstance(state, tuple):
            tensor_state, _ = state  # 如果是tuple，提取状态的张量
        else:
            tensor_state = state  # 如果不是tuple，直接使用state

        if isinstance(next_state, tuple):
            tensor_next_state, _ = next_state  # 同上
        else:
            tensor_next_state = next_state  # 同上

        # 将状态和下一个状态的tensor增加一个维度，变为批次的形式
        state_tensor = tensor_state.unsqueeze(0)
        next_state_tensor = tensor_next_state.unsqueeze(0)

        # 将状态、动作、奖励、下一状态和done标志组成一个batch
        batch = (state_tensor, action, reward, next_state_tensor, done)

        # 如果缓冲区未满，直接追加
        if len(self.buffer) < self.capacity:
            self.buffer.append(batch)
        else:
            # 如果缓冲区已满，则覆盖最旧的经验
            self.buffer[self.position] = batch

        # 更新当前经验的优先级
        if self.buffer:
            self.priorities[self.position] = self.priorities.max()  # 使用最大值作为当前优先级
        else:
            self.priorities[self.position] = 1.0  # 如果缓冲区为空，则设置优先级为1.0

        # 更新缓冲区的位置，使用循环替换老经验
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        """
        从回放缓冲区中按优先级采样一批经验。
        :param batch_size: 每次采样的批次大小
        :param beta: 用于计算重要性采样权重的beta值
        :return: 一个包含状态、动作、奖励、下一状态、done标志的batch，以及采样的权重和索引
        """
        # 计算每条经验的采样概率（优先级的alpha次方）
        if len(self.buffer) == self.capacity:
            probs = self.priorities ** self.alpha
        else:
            probs = self.priorities[:self.position] ** beta  # 使用beta来控制重要性采样的影响
        probs = probs / probs.sum()  # 归一化，使概率和为1

        # 如果概率中有NaN或Inf，替换为均匀分布
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            probs = np.ones_like(probs) / len(probs)
        # 根据计算出的概率采样经验
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # 计算每条采样经验的权重（反映重要性采样的调整）
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化权重
        weights = np.array(weights, dtype=np.float32)
        # 根据索引获取样本
        sample = [self.buffer[index] for index in indices]
        batch = list(zip(*sample))  # 打包样本，方便后续处理
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        """
        更新经验的优先级。
        :param indices: 要更新优先级的经验的索引
        :param priorities: 对应的优先级值
        """
        for index, priority in zip(indices, priorities):
            self.priorities[index] = priority  # 更新指定经验的优先级

    def __len__(self):
        """
        返回当前缓冲区中存储的经验数量。
        """
        return len(self.buffer)
