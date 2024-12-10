import os
import torch
import config
import numpy as np
import cv2
from datetime import datetime
from init_env import init_env
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent


def train(config, env, replay_buffer, agent):
    reward_history = []
    for episode in range(config.num_episodes):
        total_reward = 0.0
        state, _ = env.reset()
        win = False
        is_best = False
        while True:
            config.update(replay_buffer)
            action = agent.act(state, config.epsilon)
            next_state, reward, done, trunc, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            count = 0
            agent.replay(config, agent.training_model, agent.target_model, replay_buffer)
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", img)
            cv2.waitKey(1)
            if info.get('flag_get', False):
                win = True
            if done:
                reward_history.append(total_reward)
                recent_rewards = reward_history[-20:]
                avg_recent_reward = np.mean(recent_rewards)

                if not os.path.exists('models'):
                    os.makedirs('models')
                if total_reward > config.best_reward:
                    is_best = True

                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

                if total_reward > config.best_reward:
                    config.best_reward = total_reward
                    save_path = os.path.join('models', f'model_{config.model_counter}_{current_time}.dat')
                    config.model_counter += 1
                    torch.save(agent.training_model.state_dict(), save_path)

                print(f"Episode {episode:3d} | Reward: {total_reward:10.4f} | Win: {'Yes' if win else 'No'} | "
                      f"Epsilon: {config.epsilon:10.4f} | Avg Reward (last 20): {avg_recent_reward:10.4f} | "
                      f"Best Reward?: {is_best} | Best Reward So Far: {config.best_reward:10.4f}")
                if total_reward > 2000.0:
                    count += 1
                    save_path = os.path.join('nice_models', f'model_{count}_{current_time}.dat')
                    torch.save(agent.training_model.state_dict(), save_path)
                break


if __name__ == '__main__':
    config = config.Config(20000, 64)
    type = "training"
    env = init_env(type)
    replay_buffer = ReplayBuffer(config.buffer_size)
    agent = DQNAgent(config, (1, 84, 84), env.action_space.n)
    train(config, env, replay_buffer, agent)