import os
import torch
import config
import numpy as np
from init_env import init_env
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent


def train(config, env, replay_buffer, agent):
    """
    训练 DQN 智能体
    :param config: 配置参数（包含训练的各种超参数）
    :param env: 环境对象
    :param replay_buffer: 经验回放缓冲区
    :param agent: DQN 智能体
    """
    reward_history = []  # 存储每个回合的总奖励
    for episode in range(config.num_episodes):  # 训练过程中进行多轮（多次回合）
        total_reward = 0.0  # 每个回合的总奖励初始化
        state, _ = env.reset()  # 初始化环境并获取初始状态
        win = False  # 判断当前回合是否赢得游戏
        is_best = False  # 是否达到了历史最佳奖励
        while True:
            config.update(replay_buffer)  # 更新配置（如 epsilon 值等）
            action = agent.act(state, config.epsilon)  # 选择动作，基于当前的 epsilon-greedy 策略
            next_state, reward, done, trunc, info = env.step(action)  # 执行动作，得到下一个状态和奖励
            replay_buffer.push(state, action, reward, next_state, done)  # 将经历（状态，动作，奖励，下一状态，是否终止）推送到回放缓冲区
            state = next_state  # 更新当前状态为下一个状态
            total_reward += reward  # 累加当前回合的奖励
            agent.replay(config, agent.training_model, agent.target_model, replay_buffer)  # 执行 Q 学习中的训练步骤
            if info.get('flag_get', False):  # 判断是否达成胜利条件
                win = True
            if done:  # 如果回合结束
                reward_history.append(total_reward)  # 记录该回合的总奖励
                recent_rewards = reward_history[-20:]  # 取最近 20 个回合的奖励
                avg_recent_reward = np.mean(recent_rewards)  # 计算最近 20 个回合的平均奖励

                # 如果模型文件夹不存在，创建一个
                if not os.path.exists('models'):
                    os.makedirs('models')

                # 如果当前回合的奖励超过了历史最佳奖励，更新最佳奖励并保存模型
                if total_reward > config.best_reward:
                    is_best = True  # 标记当前回合为最佳回合
                if total_reward > config.best_reward:
                    config.best_reward = total_reward  # 更新历史最佳奖励
                    save_path = os.path.join('models', f'{"model"}_{config.model_counter}.dat')  # 生成模型保存路径
                    config.model_counter += 1  # 增加模型计数器
                    torch.save(agent.training_model.state_dict(), save_path)  # 保存模型的状态字典

                # 打印当前回合的训练进度
                print(f"Episode {episode:3d} | Reward: {total_reward:10.4f} | Win: {'Yes' if win else 'No'} | "
                      f"Epsilon: {config.epsilon:10.4f} | Avg Reward (last 20): {avg_recent_reward:10.4f} | "
                      f"Best Reward?: {is_best} | Best Reward So Far: {config.best_reward:10.4f}")
                break  # 跳出当前回合的循环，开始新的回合


# 主程序入口
if __name__ == '__main__':
    # 配置设置：经验回放缓冲区大小和批量大小
    config = config.Config(20000, 64)
    type = "training"  # 设置为训练模式
    env = init_env(type)  # 初始化环境
    replay_buffer = ReplayBuffer(config.buffer_size)  # 创建经验回放缓冲区
    agent = DQNAgent(config, (1, 84, 84), env.action_space.n)  # 创建 DQN 智能体

    # 启动训练过程
    train(config, env, replay_buffer, agent)
