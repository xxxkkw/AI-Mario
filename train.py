import pygame
import random
import numpy as np
import sys
import pygame as pg
import torch
import time
from training.dqn_model import DQNAgent
from data.main import main
from data import setup, tools
from data.states import main_menu,load_screen,level1
from data import constants as c

keybinding = {
    'action': pygame.K_s,
    'jump': pygame.K_a,
    'left': pygame.K_LEFT,
    'right': pygame.K_RIGHT,
    'down': pygame.K_DOWN
}

def train(agent, episodes=1000, max_steps=200):
    for episode in range(episodes):

        run_it = tools.Control(setup.ORIGINAL_CAPTION, agent)
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.TIME_OUT: load_screen.TimeOut(),
                      c.GAME_OVER: load_screen.GameOver(),
                      c.LEVEL1: level1.Level1()}

        run_it.setup_states(state_dict, c.MAIN_MENU)
        total_reward = 0
        while not run_it.done:
            state = run_it.get_state()
            run_it.event_loop()
            next_state = run_it.get_state()
            action = agent.act(state)
            run_it.update(action)
            next_state = run_it.get_state()
            reward = get_reward(action, c.LEVEL1)
            total_reward += reward
            agent.store_experience(state, action, reward, next_state, run_it.done)
            agent.replay()
            state = next_state
            pygame.display.update()
            if run_it.done:
                break

        print(f"Episode {episode}/{episodes} - Total Reward: {total_reward}")

        # 定期更新目标网络（根据一定的回合数）
        if episode % agent.target_update_freq == 0:
            agent.update_target_model()

        # 调整 epsilon（根据需要进行衰减）
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

def get_reward(action, level):
    current_score = level.get_score()
    score_diff = current_score - agent.previous_score
    reward = score_diff
    if action[3] == True:
        reward += 1.0
    elif action[4] == True:
        reward -= 1.0
    agent.previous_score = current_score
    return reward


# 初始化DQN智能体
agent = DQNAgent(action_size=5, state_size=(84, 84, 1), gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995)

if __name__=='__main__':
    train(agent)


