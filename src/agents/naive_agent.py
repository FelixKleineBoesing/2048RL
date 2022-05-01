import numpy as np

from src.agents.agent import Agent


class UpLeftAgent(Agent):

    def decision(self, state_space: np.ndarray) -> int:
        if self.number_decisions % 2 == 0:
            return 0
        else:
            return 2

