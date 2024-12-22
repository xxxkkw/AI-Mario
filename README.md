# 超级马里奥 DQN 智能体

该项目实现了一个使用 **DQN** 的智能体来玩 **超级马里奥**。智能体通过强化学习来最大化奖励，并学习在环境中采取最优动作。

## 特性
- **Gym Super Mario Bros 环境**：使用 gym-super-mario-bros 库创建马里奥环境。
- **DQN**：使用 PyTorch 实现 DQN 训练智能体。
- **经验回放缓冲区**：存储和采样游戏状态以稳定训练过程。
- **目标网络（Target Network）**：使用单独的目标网络来提高训练的稳定性。
- **Epsilon-贪婪策略**：在训练过程中平衡探索与利用。
## 安装
1. 克隆本仓库：
```bash
git clone https://github.com/xxxkkw/AI-Mario.git
```
2. 安装所需的依赖：
```bash
pip install -r requirements.txt
```
注：这里一定严格按照环境内的版本，要不然有bug
### 文件结构

```bash
├── models/                 # 保存的模型将存储在这里
├── train.py                # 主要训练脚本
├── agent.py                # DQN 智能体实现
├── run.py                  # 测试游戏
├── replay_buffer.py        # 经验回放缓冲区
├── config.py               # 超参数配置文件
├── init_env.py             # 马里奥环境设置和包装
├── requirements.txt        # Python 依赖库
├── final_model1-1.dat      # 模型文件
├── final_model1-2.dat      # 模型文件
└── README.md               # 项目文档
```

3. 使用前必读：
项目内置已经训练好的一个模型，可以使用
```bash
python run.py
```
来尝试玩一下已经训练好的模型。如果你想自己从头开始训练模型，只需要
```bash
python train.py
```
这样就能从头开始训练属于你的模型了.想玩别的关卡或者训练别的关卡，只需在命令行中输入时添加--level参数即可
```bash
python train.py --level 1-1
```
或者已经把某个模型训练到了一半，想继续训练，只需
```bash
python train.py --model path_to_your_model
```
此外，项目内置两个关卡的模型，1-1以及1-2，可以体验一下
```bash
python run.py --level 1-2
```
玩的开心！

[Read this in English](README-en.md)