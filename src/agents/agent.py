import abc
import numpy as np
import logging

from src.helpers import Direction


class Agent(abc.ABC):

    def __init__(self, state_shape: tuple, action_shape: tuple, name: str, caching: bool = False):
        """
        abstract class for agent which define the general interface for Agents
        :param name:
        :param side:
        """
        self.name = name
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.number_turns = 0
        self.td_loss_history = []
        self.moving_average_loss = []
        self.reward_history = []
        self.moving_average_rewards = []
        self._episode_reward = 0
        self.experience_buffer = None

        if caching:
            logging.debug("Caching is considered! When you don´t deliver cache and stream by yourself, the agent will "
                          "get a redis stream and cache by default")

    def play_turn(self, state_space: np.ndarray):
        """
        get all possible actions and decide which action to take
        :param state_space: np array describing the board
        :param action_space: dictionary containing all possible moves
        :return:
        """
        decision = self.decision(state_space)
        assert isinstance(decision, np.ndarray), "decision return must be a numpy array"
        self.number_turns += 1
        return decision

    def get_feedback(self, state: np.ndarray, action: int, reward: float, finished: bool):
        """
        through this function the agent gets information about the last turn
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param finished:
        :return: No return
        """
        if finished:
            self.reward_history.append(self._episode_reward + reward)
            self.moving_average_rewards.append(
                np.mean([self.reward_history[max([0, len(self.reward_history) - 100]):]]))
            self._episode_reward = 0
        else:
            self._episode_reward += reward
        self._get_feedback_inner(state, action, reward, finished)

    def _get_feedback_inner(self, state: np.ndarray, action: int, reward: float, finished: bool):
        """
        implement this function if you want to gather informations about your game
        :param state:
        :param action:
        :param reward:
        :param finished:
        :return:
        """
        pass

    @abc.abstractmethod
    def decision(self, state_space: np.ndarray) -> int:
        """
        this function must implement a decision based in the action_space and other delivered arguments
        return must be a dictionary with the following keys: "stone_id" and "move_index" which indicates
        the stone and move that should be executed
        :param state_space:
        :return: int: Action
        """
        pass