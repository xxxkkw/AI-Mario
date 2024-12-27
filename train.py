import os
import torch
import config
import argparse
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from init_env import init_env
from replay_buffer import ReplayBuffer
from run import test_case
from dqn_agent import DQNAgent


def train(config, env, replay_buffer, agent, mode, level):
    reward_history = []
    avg_recent_reward_list = []
    fig, ax = plt.subplots()
    ax.set_title("Average Reward Over Time (last 20 episodes)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Average Reward (last 20)")
    plt.ion()

    for episode in range(config.num_episodes):
        total_reward = 0.0
        state, _ = env.reset()
        win = False
        is_best = False
        step = 0
        while True:
            step += 1
            config.update(replay_buffer,mode)
            action = agent.act(state, config.epsilon)
            next_state, reward, done, trunc, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(config, agent.online, agent.target, replay_buffer)
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("Mario", img)
            cv2.waitKey(1)

            if info.get('flag_get', False):
                win = True
            if step % 100 == 0 and step * 0.5 > total_reward:
                done = True

            if done:
                reward_history.append(total_reward)
                recent_rewards = reward_history[-20:]
                avg_recent_reward = np.mean(recent_rewards)
                avg_recent_reward_list.append(avg_recent_reward)

                if not os.path.exists('models'):
                    os.makedirs('models')
                if total_reward > config.best_reward:
                    is_best = True
                    config.best_reward = total_reward
                    save_path = os.path.join('models',
                                             f'model_{config.model_counter}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.dat')
                    config.model_counter += 1
                    torch.save(agent.online.state_dict(), save_path)

                print(f"Episode {episode:3d} | Reward: {total_reward:10.4f} | Win: {'Yes' if win else 'No'} | "
                      f"Epsilon: {config.epsilon:10.4f} | Avg Reward (last 20): {avg_recent_reward:10.4f} | "
                      f"Best Reward?: {is_best} | Best Reward So Far: {config.best_reward:10.4f}")

                if episode % 20 == 0 or win:
                    if test_case(agent, level):
                        print("test pass")
                        final_model_path = f'final_model_{int(total_reward)}.dat'
                        torch.save(agent.online.state_dict(), final_model_path)
                    else:
                        print("test failed")

                ax.clear()
                ax.plot(avg_recent_reward_list, label="Average Reward (last 20)")
                ax.set_title("Average Reward Over Time (last 20 episodes)")
                ax.set_xlabel("Episodes")
                ax.set_ylabel("Average Reward (last 20)")
                ax.legend()
                plt.pause(0.01)

                break
    plt.ioff()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="Train the DQN agent with a pre-trained model.")
    parser.add_argument("--model", type=str, default=None, help="Path to the pre-trained model")
    parser.add_argument("--level", type=str,default="1-1", help="Level number for the model, such as 1-1, 1-2")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = config.Config(20000, 64)
    env = init_env(args.level)
    replay_buffer = ReplayBuffer(config.buffer_size)
    pre_train_model = args.model
    if pre_train_model:
        mode = "pretrain"
    else:
        mode = "train"
    agent = DQNAgent(config, (4, 84, 84), env.action_space.n, pre_train_model)
    train(config, env, replay_buffer, agent, mode, args.level)

