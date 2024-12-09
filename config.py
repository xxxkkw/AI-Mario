import math


class Config:
    def __init__(self, buffer_size, batch_size):
        self.best_reward = -float('inf')
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_times = 10000
        self.training_episodes = 0
        self.num_episodes = 50000
        self.model_counter = 0
        self.epsilon = 0.99
        self.beta = 0.5

    def training_episode(self):
        return self.training_episodes

    def learning_times(self):
        return self.learning_times

    def model_counter(self):
        self.model_counter += 1
        return self.model_counter

    def num_episodes(self):
        return self.num_episodes

    def best_reward(self):
        return self.best_reward

    def buffer_size(self):
        return self.buffer_size

    def batch_size(self):
        return self.batch_size

    def epsilon(self):
        return self.epsilon

    def beta(self):
        return self.beta

    def update(self, replay_buffer):
        self.training_episodes += 1
        if len(replay_buffer) > self.batch_size:
            self.beta = min(1.0, 0.5 + self.training_episodes * 6e-5)
        else:
            self.beta = 0.5
        self.epsilon = 0.1 + 0.99 * math.exp(-1.0 * ((self.training_episodes + 1) / 100000))
