import pygame
import config
from training.dqn_model import DQNAgent
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

def train(agent):
    run_it = tools.Control(setup.ORIGINAL_CAPTION, agent)
    MENU = main_menu.Menu()
    LOAD = load_screen.LoadScreen()
    TIMEOUT = load_screen.TimeOut()
    GAMEOVER = load_screen.GameOver()
    LEVEL = level1.Level1()
    state_dict = {c.MAIN_MENU: MENU,
                c.LOAD_SCREEN: LOAD,
                c.TIME_OUT: TIMEOUT,
                c.GAME_OVER: GAMEOVER,
                c.LEVEL1: LEVEL}

    run_it.setup_states(state_dict, c.MAIN_MENU)
    total_reward = 0
    repeat_action_count = 0
    state = run_it.get_state()
    action = None
    while not run_it.done:
        run_it.event_loop()
        if repeat_action_count == 0:
            repeat_action_count = 30
            action = agent.act(state)
        repeat_action_count -= 1
        action_key = [False] * 5
        action_key[action] = True
        run_it.update(action_key)  # 更新状态
        next_state = run_it.get_state()
        reward = get_reward(action, agent)
        total_reward += reward
        agent.store_experience(state, action, reward, next_state, run_it.done)
        state = next_state
        pygame.display.update()
        run_it.clock.tick(60)
        if config.dead:
            print(f"Episode {config.episodes}/{2000} - Total Reward: {total_reward}")
            print("random count: ", config.random_count)
            print("model count: ", config.model_count)
            print("model percentage", config.model_count / (config.model_count + config.random_count))
            agent.replay()
            # 定期更新目标网络（根据一定的回合数）
            if config.episodes % agent.target_update_freq == 0:
                agent.update_target_model()
                # 调整 epsilon（根据需要进行衰减）
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
                agent.epsilon = max(agent.epsilon, agent.epsilon_min)
            config.total_score = 0
            config.random_count = 0
            config.model_count = 0
            total_reward = 0
            config.dead = False

def get_reward(action, agent):
    current_score = config.total_score
    score_diff = current_score - agent.previous_score
    reward = score_diff
    if action == 3:
        reward += 1
    if config.dead:
        reward -= 1000
    agent.previous_score = current_score
    return reward

if __name__=='__main__':
    agent = DQNAgent(action_size=5, state_size=(84, 84, 1), gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                     epsilon_decay=0.995)
    train(agent)


