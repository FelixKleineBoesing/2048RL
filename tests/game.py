import unittest
import numpy as np

from src.game import Game, Direction


class BoardTester(unittest.TestCase):

    def _setup_deterministic_board(self, board: Game, numpy_array: np.ndarray):
        board.board = numpy_array
        return board

    def init_board(self):
        start = np.array([
            [0, 2, 0, 2],
            [0, 0, 0, 0],
            [2, 0, 2, 2],
            [4, 0, 2, 0]
        ])
        board = Game(4)
        board = self._setup_deterministic_board(board, start)
        return board

    def test_tile_moving_down(self):
        board = self.init_board()
        number_rotations = board._get_number_rotations(Direction.DOWN)
        board._rotate_board(number_rotations)
        board._move_tiles_to_edge()
        board._rotate_board(4 - number_rotations)
        actual_result = board.board

        expected_result = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [2, 0, 2, 2],
                                    [4, 2, 2, 2]])
        self.assertTrue((expected_result == actual_result).all())

    def test_tile_moving_up(self):
        board = self.init_board()
        number_rotations = board._get_number_rotations(Direction.UP)
        board._rotate_board(number_rotations)
        board._move_tiles_to_edge()
        board._rotate_board(4 - number_rotations)
        actual_result = board.board

        expected_result = np.array([[2, 2, 2, 2],
                                    [4, 0, 2, 2],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]])
        self.assertTrue((expected_result == actual_result).all())

    def test_tile_moving_left(self):
        board = self.init_board()
        number_rotations = board._get_number_rotations(Direction.LEFT)
        board._rotate_board(number_rotations)
        board._move_tiles_to_edge()
        board._rotate_board(4 - number_rotations)
        actual_result = board.board

        expected_result = np.array([[2, 2, 0, 0],
                                    [0, 0, 0, 0],
                                    [2, 2, 2, 0],
                                    [4, 2, 0, 0]])
        self.assertTrue((expected_result == actual_result).all())

    def test_tile_moving_right(self):
        board = self.init_board()
        number_rotations = board._get_number_rotations(Direction.RIGHT)
        board._rotate_board(number_rotations)
        board._move_tiles_to_edge()
        board._rotate_board(4-number_rotations)
        actual_result = board.board

        expected_result = np.array([[0, 0, 2, 2],
                                    [0, 0, 0, 0],
                                    [0, 2, 2, 2],
                                    [0, 0, 4, 2]])
        self.assertTrue((expected_result == actual_result).all())

    def test_tile_merge_up(self):
        board = self.init_board()
        board.make_move(Direction.UP, spawn_new=False)
        actual_result = board.board
        expected_result = np.array([
            [2, 2, 4, 4],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.assertTrue((expected_result == actual_result).all())

    def test_tile_merge_down(self):
        board = self.init_board()
        board.make_move(Direction.DOWN, spawn_new=False)
        actual_result = board.board
        expected_result = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [4, 2, 4, 4]
        ])
        self.assertTrue((expected_result == actual_result).all())

    def test_tile_merge_left(self):
        board = self.init_board()
        board.make_move(Direction.LEFT, spawn_new=False)
        actual_result = board.board
        expected_result = np.array([
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 2, 0, 0],
            [4, 2, 0, 0]
        ])
        self.assertTrue((expected_result == actual_result).all())

    def test_tile_merge_right(self):
        board = self.init_board()
        board.make_move(Direction.RIGHT, spawn_new=False)
        actual_result = board.board
        expected_result = np.array([
            [0, 0, 0, 4],
            [0, 0, 0, 0],
            [0, 0, 2, 4],
            [0, 0, 4, 2]
        ])
        self.assertTrue((expected_result == actual_result).all())
