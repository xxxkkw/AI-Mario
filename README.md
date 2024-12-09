# 超级马里奥 DQN 智能体

该项目实现了一个使用 **DQN** 的智能体来玩 ** 超级马里奥 ** 游戏。智能体通过强化学习来最大化奖励，并学习在环境中采取最优动作。

## 特性
- **Gym Super Mario Bros 环境**：使用 `gym-super-mario-bros` 库创建马里奥环境。
- **DQN**：使用 PyTorch 实现 DQN 训练智能体。
- **经验回放缓冲区**：存储和采样游戏状态以稳定训练过程。
- **目标网络（Target Network）**：使用单独的目标网络来提高训练的稳定性。
- **Epsilon-贪婪策略**：在训练过程中平衡探索与利用。

## 安装

1. 克隆本仓库：
   ```bash
   https://github.com/xxxkkw/AI-Mario.git
   ```

2.安装所需的依赖：
  ```
  pip install -r requirements.txt
  ```
注：这里一定严格按照环境内的版本，要不然有bug

## 文件结构
```bash

├── models/                 # 保存的模型将存储在这里
├── src/
│   ├── train.py            # 主要训练脚本
│   ├── agent.py            # DQN 智能体实现
│   ├── replay_buffer.py    # 经验回放缓冲区
│   ├── config.py           # 超参数配置文件
│   └── init_env.py         # 马里奥环境设置和包装
├── requirements.txt        # Python 依赖库
└── README.md               # 项目文档
```



