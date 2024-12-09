import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.alpha = 0.6
        self.beta = 0.5
        self.capacity = capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if isinstance(state, tuple):
            tensor_state, _ = state
        else:
            tensor_state = state
        if isinstance(next_state, tuple):
            tensor_next_state, _ = next_state
        else:
            tensor_next_state = next_state
        state_tensor = tensor_state.unsqueeze(0)
        next_state_tensor = tensor_next_state.unsqueeze(0)
        batch = (state_tensor, action, reward, next_state_tensor, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(batch)
        else:
            self.buffer[self.position] = batch

        if self.buffer:
            self.priorities[self.position] = self.priorities.max()
        else:
            self.priorities[self.position] = 1.0

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            probs = self.priorities ** self.alpha
        else:
            probs = self.priorities[:self.position] ** beta
        probs = probs / probs.sum()
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            probs = np.ones_like(probs) / len(probs)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = np.array(weights, dtype=np.float32)
        sample = [self.buffer[index] for index in indices]
        batch = list(zip(*sample))
        return batch, indices, weights


    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.priorities[index] = priority

    def __len__(self):
        return len(self.buffer)
