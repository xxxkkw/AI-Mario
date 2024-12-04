from . import setup, tools
from .states import main_menu, load_screen, level1
from . import constants as c
from training.dqn_model import DQNAgent

def main():
    """Add states to control here."""
    agent = DQNAgent(action_size=5, state_size=(84, 84, 1), gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                     epsilon_decay=0.995)
    run_it = tools.Control(setup.ORIGINAL_CAPTION, agent)
    state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.TIME_OUT: load_screen.TimeOut(),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.LEVEL1: level1.Level1()}

    run_it.setup_states(state_dict, c.MAIN_MENU)
    run_it.main()



