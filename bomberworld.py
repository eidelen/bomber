# bomberworld.py
# Author: Adrian Schneider, armasuisse
# Note: Initial copied from Giacomo Del Rio, IDSIA

from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
import copy
from random import randrange

# Best performance when size = 10 and no penalty on moving and allowed being close to bomb
# 10 x 10 = 100 stones - 6 = 94
# bombs necessary = 94 / 5 = 19 bombs
# reward = 94 + 10 - 19 = about 85

class BomberworldEnv(gym.Env):

    def __init__(self, size: int | List[int], max_steps: int, indestructible_agent=True, dead_near_bomb=False, dead_when_colliding=False, reduced_obs=False, move_penalty=-0.2, collision_penalty=-1.0,
                 bomb_penalty=-1.0, close_bomb_penalty=-2.0, rock_reward=1.0, end_game_reward=10.0 ):
        """
        Parameters
        ----------
        size: Board size
        max_steps: Max steps in one game
        indestructible_agent: If True, bomb explodes immediately and agent is indestructible by own bomb.
                              If False, bomb explodes 2 steps later and agent needs to be in safe distance.
        dead_near_bomb: Only active when indestructible_agent==False. When true, game ends when agent too close to bomb.
        dead_when_colliding: When true, game ends when agent collides with rock or wall.
        reduced_obs: When true, agent only sees surrounding 3x3 patch.
        reward / penalty: Several reward and penalty options.
        """

        # settings
        self.rock_val = 0.0
        self.agent_val = 1.0
        self.bomb_val = 0.25
        self.bomb_and_agent_val = 0.75
        self.empty_val = 0.5

        self.move_penalty = move_penalty
        self.collision_penalty = collision_penalty
        self.bomb_penalty = bomb_penalty
        self.close_bomb_penalty = close_bomb_penalty
        self.rock_reward = rock_reward
        self.end_game_reward = end_game_reward

        self.size = size
        self.board_size = None
        self.max_steps = max_steps
        self.indestructible_agent = indestructible_agent
        self.dead_near_bomb = dead_near_bomb
        self.dead_when_colliding = dead_when_colliding
        self.reduced_obs = reduced_obs
        self.current_step = 0

        if self.reduced_obs:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3 * 3,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(size * size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0

        if type(self.size) is list: # randomly select a board size form the list
            self.board_size = self.size[randrange(len(self.size))]

            # normalize penalties and rewards relative to first size in list
            main_size = self.size[0]

            # reward = (total reward with main_size) / board_size
            self.current_move_penalty = (self.move_penalty * (main_size ** 2)) / (self.board_size ** 2)
            self.current_collision_penalty = (self.collision_penalty * (main_size ** 2)) / (self.board_size ** 2)
            self.current_bomb_penalty = (self.bomb_penalty * (main_size ** 2)) / (self.board_size ** 2)
            self.current_close_bomb_penalty = (self.close_bomb_penalty * (main_size ** 2)) / (self.board_size ** 2)
            self.current_rock_reward = (self.rock_reward * (main_size ** 2)) / (self.board_size ** 2)
            self.current_max_steps = (self.max_steps / (main_size ** 2)) * (self.board_size ** 2) # increase with board size
            self.current_end_game_reward = self.end_game_reward # endgame reward independant of board size
        else:
            self.board_size = self.size
            self.current_move_penalty = self.move_penalty
            self.current_collision_penalty = self.collision_penalty
            self.current_bomb_penalty = self.bomb_penalty
            self.current_close_bomb_penalty = self.close_bomb_penalty
            self.current_rock_reward = self.rock_reward
            self.current_max_steps = self.max_steps
            self.current_end_game_reward = self.end_game_reward  # endgame reward independant of board size

        self.set_initial_board(self.board_size, tuple(self.np_random.integers(low=0, high=self.board_size, size=2)))
        return self.make_observation(), {}

    def set_initial_board(self, size, agent_pos):
        self.stones = np.full((size, size), True)
        self.agent_pos = agent_pos
        self.active_bombs = []

        # initially remove all 8 stones around the agent
        self.bomb_3x3(agent_pos)

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        m, n = pos
        return (-1 < m < self.board_size) and (-1 < n < self.board_size)

    def can_move_to_pos(self, pos: Tuple[int, int]) -> bool:
        return self.is_valid_pos(pos) and (not self.stones[pos])

    def make_current_board_2D(self) -> np.ndarray:
        board = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        # set rocks
        for m, n in np.ndindex(self.stones.shape):
            board[(m, n)] = self.rock_val if self.stones[(m, n)] else self.empty_val
        # set active bombs
        for bomb_pos, _ in self.active_bombs:
            board[bomb_pos] = self.bomb_val
        # set agent
        board[self.agent_pos] = self.bomb_and_agent_val if self.is_active_bomb_on_field(
            self.agent_pos) else self.agent_val

        return board

    def make_observation_2D(self) -> np.ndarray:
        board = self.make_current_board_2D()
        if self.reduced_obs: # cut 3x3 patch around agent
            m_ap, n_ap = self.agent_pos
            m_center = max(1, m_ap)
            m_center = min(self.board_size - 2, m_center)
            n_center = max(1, n_ap)
            n_center = min(self.board_size - 2, n_center)
            return board[m_center-1:m_center+2, n_center-1:n_center+2]
        else:
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
        agent_killed = False

        # debug info
        placed_bomb = None
        exploded_bomb = None

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
                reward += self.current_move_penalty # penalty for each move
            else:
                reward += self.current_collision_penalty
                if self.dead_when_colliding:
                    agent_killed = True

        elif action == 4: # drop bomb at agent location
            reward += self.current_bomb_penalty  # penalty for each dropped bomb
            placed_bomb = self.agent_pos
            if self.indestructible_agent:
                self.active_bombs.append((self.agent_pos, 0)) # immediate detonation
            else:
                self.active_bombs.append((self.agent_pos, 2))  # detonation two steps later

        #go through all active bombs
        still_active_bombs = []
        for bomb_pos, step_timer in self.active_bombs:
            if step_timer <= 0:
                reward += self.current_rock_reward * self.bomb_3x3(bomb_pos) # detonate bomb
                exploded_bomb = bomb_pos

                if not self.indestructible_agent:
                    # check that agent is in safe distance
                    squared_dist = (bomb_pos[0]-self.agent_pos[0])**2 + (bomb_pos[1]-self.agent_pos[1])**2
                    if squared_dist < 4.0:
                        reward += self.current_close_bomb_penalty
                        if self.dead_near_bomb:
                            agent_killed = True
            else:
                still_active_bombs.append((bomb_pos, step_timer - 1))

        self.active_bombs = still_active_bombs

        # mission completed when every rock was bombed
        if (self.stones == False).all():
            reward += self.current_end_game_reward
            terminated = True
        else:
            terminated = False

        if self.current_step > self.current_max_steps or agent_killed: # end game when max step reached or agent killed
            truncate = True
        else:
            truncate = False

        self.current_step += 1

        return self.make_observation(), reward, terminated, truncate, {"placed_bomb": placed_bomb, "exploded_bomb": exploded_bomb}
