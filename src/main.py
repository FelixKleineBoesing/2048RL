from src.agents.naive_agent import UpLeftAgent
from src.agents.random import RandomAgent
from src.game import Env


def main():
    env = Env()
    action_space = env.get_action_space()
    state_space = env.get_state_space()
    agent = RandomAgent(state_shape=state_space, action_shape=action_space, name="RandomAgent")
    env.assign_agent(agent)
    env.run_multiple_games(1000)
    env.create_histogram_of_point_history()


    env = Env()
    up_left_agent = UpLeftAgent(state_shape=state_space, action_shape=action_space, name="UpLeftAgent")
    env.assign_agent(up_left_agent)
    env.run_multiple_games(1000)
    env.create_histogram_of_point_history()

if __name__ == "__main__":
    main()