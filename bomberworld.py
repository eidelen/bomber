# bomberworld.py
# Initial copied from Giacomo Del Rio, IDSIA

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import copy

# Best performance when size = 10 and no penalty on moving and being close to bomb
# 10 x 10 = 100 stones - 6 = 94
# bombs necessary = 94 / 5 = 19 bombs
# reward = 94 + 10 - 19 = about 85

class BomberworldEnv(gym.Env):

    def __init__(self, size: int, max_steps: int, indestructible_agent=True):
        """
        Parameters
        ----------
        size: Board size
        max_steps: Max steps in one game
        indestructible_agent: If True, bomb explodes immediately and agent is indestructible by own bomb.
                              If False, bomb explodes 2 steps later and agent needs to be in safe distance.
        """


        # settings
        self.rock_val = 0.0
        self.agent_val = 1.0
        self.bomb_val = 0.25
        self.bomb_and_agent_val = 0.75
        self.empty_val = 0.5

        self.move_penalty = -0.2
        self.collision_penalty = -1.0
        self.bomb_penalty = -1.0
        self.close_bomb_penalty = -2.0
        self.rock_reward = 1.0
        self.end_game_reward = 10.0

        self.size = size
        self.max_steps = max_steps
        self.indestructible_agent = indestructible_agent
        self.current_step = 0

        self.agent_pos = (0, 0)
        self.stones = np.full((self.size, self.size), True)
        self.active_bombs = []

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(size * size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.set_initial_board(tuple(self.np_random.integers(low=0, high=self.size, size=2)))
        return self.make_observation(), {}

    def set_initial_board(self, agent_pos):
        self.stones = np.full((self.size, self.size), True)
        self.agent_pos = agent_pos

        # initially remove all 8 stones around the agent
        self.bomb_3x3(agent_pos)

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        m, n = pos
        return (-1 < m < self.size) and (-1 < n < self.size)

    def can_move_to_pos(self, pos: Tuple[int, int]) -> bool:
        return self.is_valid_pos(pos) and (not self.stones[pos])

    def make_observation_2D(self) -> np.ndarray:
        board = np.zeros((self.size, self.size), dtype=np.float32)
        # set rocks
        for m, n in np.ndindex(self.stones.shape):
            board[(m, n)] = self.rock_val if self.stones[(m, n)] else self.empty_val
        # set active bombs
        for bomb_pos, _ in self.active_bombs:
            board[bomb_pos] = self.bomb_val
        # set agent
        board[self.agent_pos] = self.bomb_and_agent_val if self.is_active_bomb_on_field(self.agent_pos) else self.agent_val
        return board

    def make_observation(self) -> np.ndarray:
        o = self.make_observation_2D()
        return o.flatten()

    def bomb_3x3(self, pos: Tuple[int, int]) -> int:
        pm, pn = pos

        n_bombed = 0

        for m in range(pm - 1, pm + 2):
            for n in range(pn - 1, pn + 2):
                if self.is_valid_pos((m, n)) and self.stones[(m, n)]:
                    self.stones[(m, n)] = False
                    n_bombed += 1

        return n_bombed


    def is_active_bomb_on_field(self, pos: Tuple[int, int]) -> bool:
        for bomb_pos, _ in self.active_bombs:
            if bomb_pos == pos:
                return True
        return False


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:

        reward = 0.0

        if action < 4: # move actions
            if action == 0: # up
                next_pos = (self.agent_pos[0]-1, self.agent_pos[1])
            elif action == 1: # right
                next_pos = (self.agent_pos[0], self.agent_pos[1]+1)
            elif action == 2:  # down
                next_pos = (self.agent_pos[0]+1, self.agent_pos[1])
            elif action == 3:  # left
                next_pos = (self.agent_pos[0], self.agent_pos[1]-1)

            if self.can_move_to_pos(next_pos):
                self.agent_pos = next_pos
                reward += self.move_penalty # penalty for each move
            else:
                reward += self.collision_penalty

        elif action == 4: # drop bomb at agent location
            reward += self.bomb_penalty  # penalty for each dropped bomb
            if self.indestructible_agent:
                self.active_bombs.append((self.agent_pos, 0)) # immediate detonation
            else:
                self.active_bombs.append((self.agent_pos, 2))  # detonation two steps later

        #go through all active bombs
        still_active_bombs = []
        for bomb_pos, step_timer in self.active_bombs:
            if step_timer <= 0:
                reward += self.rock_reward * self.bomb_3x3(bomb_pos) # detonate bomb

                if not self.indestructible_agent:
                    # check that agent is in safe distance
                    squared_dist = (bomb_pos[0]-self.agent_pos[0])**2 + (bomb_pos[1]-self.agent_pos[1])**2
                    if squared_dist < 4.0:
                        reward += self.close_bomb_penalty
            else:
                still_active_bombs.append((bomb_pos, step_timer - 1))

        self.active_bombs = still_active_bombs

        # mission completed when every rock was bombed
        if (self.stones == False).all():
            reward += self.end_game_reward
            terminated = True
        else:
            terminated = False

        if self.current_step > self.max_steps:
            truncate = True
        else:
            truncate = False

        self.current_step += 1

        return self.make_observation(), reward, terminated, truncate, {}
