import os
import torch
import config
import numpy as np
import cv2
from datetime import datetime
from init_env import init_env
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent


# 训练函数
def train(config, env, replay_buffer, agent):
    reward_history = []  # 用于记录奖励历史，便于监控性能
    # 遍历每一集训练
    for episode in range(config.num_episodes):
        total_reward = 0.0  # 初始化该集的总奖励
        state, _ = env.reset()  # 重置环境并获得初始状态
        win = False  # 用于跟踪智能体是否赢得了本集
        is_best = False  # 用于标记本集的奖励是否是迄今为止的最好奖励
        count = 1
        # 直到本集结束
        while True:
            # 更新配置，可能会影响Replay Buffer
            config.update(replay_buffer)
            # 基于当前状态和epsilon-贪心策略选择动作
            action = agent.act(state, config.epsilon)
            # 执行动作，并获取下一个状态、奖励和终止标志
            next_state, reward, done, trunc, info = env.step(action)
            # 将当前经验存储到Replay Buffer中
            replay_buffer.push(state, action, reward, next_state, done)
            # 更新状态为下一个状态
            state = next_state
            # 累加奖励
            total_reward += reward
            # 进行智能体的经验回放（训练模型）
            agent.replay(config, agent.training_model, agent.target_model, replay_buffer)
            # 渲染环境并显示当前帧（用于可视化）
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", img)
            cv2.waitKey(1)
            # 检查智能体是否到达目标或赢得了比赛
            if info.get('flag_get', False):
                win = True
            # 如果集结束（done），则退出循环
            if done:
                # 记录本集的总奖励
                reward_history.append(total_reward)
                # 获取最近20集的平均奖励
                recent_rewards = reward_history[-20:]
                avg_recent_reward = np.mean(recent_rewards)
                # 确保'models'文件夹存在，用于保存模型
                if not os.path.exists('models'):
                    os.makedirs('models')
                # 检查当前集的奖励是否是最好的
                if total_reward > config.best_reward:
                    is_best = True
                # 获取当前时间戳，用于命名模型文件
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 如果当前集的奖励高于最佳奖励，则保存模型
                if total_reward > config.best_reward:
                    config.best_reward = total_reward
                    save_path = os.path.join('models', f'model_{config.model_counter}_{current_time}.dat')
                    config.model_counter += 1
                    torch.save(agent.training_model.state_dict(), save_path)
                # 输出本集的训练统计信息
                print(f"Episode {episode:3d} | Reward: {total_reward:10.4f} | Win: {'Yes' if win else 'No'} | "
                      f"Epsilon: {config.epsilon:10.4f} | Avg Reward (last 20): {avg_recent_reward:10.4f} | "
                      f"Best Reward?: {is_best} | Best Reward So Far: {config.best_reward:10.4f}")
                # 如果智能体的奖励超过2000，则保存模型
                if total_reward > 2000.0:
                    count += 1
                    save_path = os.path.join('nice_models', f'model_{count}_{current_time}.dat')
                    torch.save(agent.training_model.state_dict(), save_path)
                break  # 结束当前集


# 主函数，执行训练
if __name__ == '__main__':
    # 初始化配置，指定训练集数量和Replay Buffer的大小
    config = config.Config(20000, 64)
    # 定义环境类型（"training"表示一种特定的环境设置）
    type = "training"
    # 初始化环境
    env = init_env(type)
    # 初始化Replay Buffer，指定缓冲区大小
    replay_buffer = ReplayBuffer(config.buffer_size)
    # 初始化DQN智能体，指定配置、输入状态形状和动作空间大小
    agent = DQNAgent(config, (1, 84, 84), env.action_space.n)
    # 开始训练过程
    train(config, env, replay_buffer, agent)
