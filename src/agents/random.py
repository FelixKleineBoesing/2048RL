import numpy as np

from src.agents.agent import Agent


class RandomAgent(Agent):

    def decision(self, state_space: np.ndarray) -> int:
        return int(np.random.choice([0, 1, 2, 3]))