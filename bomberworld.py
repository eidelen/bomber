# bomberworld.py
# Initial copied from Giacomo Del Rio, IDSIA

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class GridworldEnv(gym.Env):

    def __init__(self, size: int, max_steps: int):
        self.size = size
        self.max_steps = max_steps
        self.agent_pos = (0, 0)
        self.current_step = 0
        self.board = np.zeros(shape=(self.size, self.size), dtype=np.uint8)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size, size, 1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.board = np.zeros(shape=(self.size, self.size), dtype=np.uint8)
        self.agent_pos = tuple(self.np_random.integers(low=0, high=self.size, size=2))

        self.current_step = 0

        return self.make_observation(), {}

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        m, n = pos
        return (-1 < m < self.size) and (-1 < n < self.size)

    def can_move_to_pos(self, pos: Tuple[int, int]) -> bool:
        return self.is_valid_pos(pos) and self.board[pos] > 0

    def make_observation(self) -> np.ndarray:
        o = np.zeros(shape=(self.size, self.size), dtype=np.uint8)
        o[self.agent_pos] = 255
        return o #o.reshape((self.size, self.size, 1))

    def bomb_3x3(self, pos: Tuple[int, int]) -> int:
        pm, pn = pos

        n_bombed = 0

        for m in range(pm - 1, pm + 2):
            for n in range(pn - 1, pn + 2):
                if self.is_valid_pos((m, n)) and self.board[(m, n)] == 0:
                    self.board[(m, n)] = 128
                    n_bombed += 1

        return n_bombed


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:

        reward = 0

        if action == 0: # up
            if self.agent_pos[0] == 0:
                reward = -1
            else:
                self.agent_pos = (self.agent_pos[0]-1, self.agent_pos[1])
        elif action == 1: # right
            if self.agent_pos[1] == self.size - 1:
                reward = -1
            else:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == 2:  # down
            if self.agent_pos[0] == self.size - 1:
                reward = -1
            else:
                self.agent_pos = (self.agent_pos[0]+1, self.agent_pos[1])
        elif action == 3:  # left
            if self.agent_pos[1] == 0:
                reward = -1
            else:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)

        if self.agent_pos == self.goal_pos:
            reward = 10
            terminated = True
        else:
            reward = reward - 0.1 # punish each move which does not reach the target
            terminated = False

        if self.current_step > self.max_steps:
            truncate = True
        else:
            truncate = False

        self.current_step += 1

        return self.make_observation(), reward, terminated, truncate, {}
