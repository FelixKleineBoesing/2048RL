from math import log
import pandas as pd
import numpy as np
import logging
import plotly.express as px

from src.agents.agent import Agent
from src.agents.naive_agent import UpLeftAgent
from src.agents.random import RandomAgent
from src.helpers import Direction


class Game:

    def __init__(self, number_tiles: int = 4):
        self.points = 0
        self.number_tiles = number_tiles
        self.board = np.zeros((number_tiles, number_tiles), dtype=int)
        self._empty_tiles = {(i, j) for i in range(number_tiles)
                             for j in range(number_tiles)}
        self._generate_tile_and_assign_to_board()
        self._generate_tile_and_assign_to_board()

    def _generate_tile_and_assign_to_board(self):
        """
        Generates a new tile on the board by randomly choosing from the
        possible values of 2 or 4.
        :return bool: True if the board is not full, False otherwise.
        """
        if len(self._empty_tiles) == 0:
            return False
        tile_value = np.random.choice([2, 4], p=[0.9, 0.1])
        tile_position_id = np.random.choice(len(self._empty_tiles))
        tile_position = list(self._empty_tiles)[tile_position_id]
        self.board[tile_position] = tile_value
        self._empty_tiles.discard(tile_position)
        return True

    def make_move(self, direction: Direction, spawn_new: bool = True):
        """

        :param direction:
        :return:
        """
        number_rotations = self._get_number_rotations(direction)
        self._rotate_board(number_rotations=number_rotations)
        self._merge_tiles()
        self._rotate_board(number_rotations=4-number_rotations)
        self._get_empty_tiles()
        if len(self._empty_tiles) == 0:
            return "Game Over"
        if spawn_new:
            self._generate_tile_and_assign_to_board()
        return "Success"

    def _get_empty_tiles(self):
        self._empty_tiles = {(i, j) for i in range(self.number_tiles)
                             for j in range(self.number_tiles) if self.board[i, j] == 0}

    def _get_number_rotations(self, direction: Direction):
        if direction == Direction.UP:
            number_rotations = 2
        elif direction == Direction.DOWN:
            number_rotations = 0
        elif direction == Direction.LEFT:
            number_rotations = 1
        elif direction == Direction.RIGHT:
            number_rotations = 3
        else:
            raise ValueError("Invalid direction")
        return number_rotations

    def _move_tiles_to_edge(self):
        for i in range(self.number_tiles - 1):
            for j in range(self.number_tiles):
                if self.board[i+1, j] == 0 and self.board[i, j] != 0:
                    move_values = self.board[:i+1, j].copy()
                    self.board[:i+1, j] = 0
                    self.board[(i+2-len(move_values)):(i+2), j] = move_values

    def _rotate_board(self, number_rotations: int):
        if number_rotations > 0:
            self.board = np.rot90(self.board, number_rotations)

    def _add_points(self, merged_tile_value: int):
        self.points += merged_tile_value * 2

    def _merge_tiles(self):
        """

        :param dim_range: tuple of start and end indices of the dimension
        :param axis: 0 for rows, 1 for columns
        :return:
        """
        self._move_tiles_to_edge()
        for i in range(self.number_tiles - 1, 0, -1):
            for j in range(self.number_tiles):
                if self.board[i, j] == self.board[i-1, j] and self.board[i, j] != 0:
                    self._add_points(self.board[i, j])
                    self.board[i, j] = self.board[i, j] * 2
                    move_values = self.board[:i-1, j].copy()
                    self.board[:i, j] = 0
                    self.board[(i-len(move_values)):i, j] = move_values


class Env:

    def __init__(self, number_tiles: int = 4, max_steps_per_game: int = 500, max_value: int = 8192,
                 flattened_state: bool = False):
        self.max_steps_per_game = max_steps_per_game
        self.number_tiles = number_tiles
        self.max_value = max_value
        self.number_powers = int(log(8192, 2)) + 1
        self.possible_numbers = [0] + [2 ** i for i in range(1, self.number_powers)]
        self.agent = None
        self.point_history = []
        self.flattened_state = flattened_state
        self._init_game()

    def assign_agent(self, agent: Agent):
        self.agent = agent

    def _init_game(self):
        self.game = Game(number_tiles=self.number_tiles)

    def get_action_space(self):
        return (4, )

    def get_state_space(self):
        return (self.number_tiles ** 2 * self.number_powers, )

    def do_action(self, action: int) -> (int,  int, np.ndarray, bool):
        """

        :param action:
        :return: Reward, Action, State, finished
        """
        points_before = self.game.points
        status = self.game.make_move(direction=Direction(action))
        if status == "Game Over":
            is_finished = True
        else:
            is_finished = False
        points_after = self.game.points
        return points_after - points_before, action, self.game.board, is_finished

    def run(self):
        if self.agent is None:
            raise ValueError("Agent must be assigned first")
        finished = False
        while not finished:
            s = self.game.board
            action = self.agent.play_turn(s)
            reward, action, state, is_finished = self.do_action(action)
            if self.flattened_state:
                state = state.reshape((self.number_tiles * self.number_tiles, ))
                state = self._get_dummies(state)
                state = state.reshape(self.number_tiles*self.number_tiles*self.number_powers)
            if is_finished:
                self.point_history.append(self.game.points)
                break

            self.agent.get_feedback(state=state, action=action, reward=reward, finished=is_finished)

    def _get_dummies(self, state):
        data = np.zeros((self.number_tiles * self.number_tiles, self.number_powers))
        for i, val in enumerate(self.possible_numbers):
            data[:, i] = state == val
        return data

    def run_multiple_games(self, number_games: int):
        for game in range(number_games):
            logging.info(f"Running Game Number: {game}")
            self._init_game()
            self.run()

    def create_histogram_of_point_history(self):
        fig = px.histogram(self.point_history)
        fig.show()
    