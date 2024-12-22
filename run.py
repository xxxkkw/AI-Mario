from dqn_agent import DQNAgent
from init_env import init_env
import argparse
import config
import time
import cv2


def test_case(agent, level):
    type = "running"
    env = init_env(level)
    state, _ = env.reset()
    done = False
    win = False
    while not done:
        img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        cv2.imshow("Mario", img)
        cv2.waitKey(1)
        action = agent.agent_act(state)
        state, reward, done, trunc, info = env.step(action)
        if info.get('flag_get', False):
            win = True
    return win


def run_model(level):
    env = init_env(level)
    state, _ = env.reset()
    done = False
    win = False
    model_path = f"final_model{level}.dat"
    agent = DQNAgent(config, (4, 84, 84), env.action_space.n, model_path)
    target_fps = 60
    frame_delay = 1 / target_fps
    while not done:
        start_time = time.time()
        img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        cv2.imshow("Mario", img)
        cv2.waitKey(int(frame_delay * 1000))
        action = agent.agent_act(state)
        state, reward, done, trunc, info = env.step(action)
        if info.get('flag_get', False):
            print("test pass")

        frame_processing_time = time.time() - start_time
        if frame_processing_time < frame_delay:
            time.sleep(frame_delay - frame_processing_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model with specified level.")
    parser.add_argument("--level", type=str, default="1-1", help="Level number for the model, such as 1-1, 1-2")
    args = parser.parse_args()
    run_model(args.level)
