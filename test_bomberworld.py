import math
import unittest
import numpy as np
import bomberworld
from bomberworld_plotter import BomberworldPlotter

class MyTestCase(unittest.TestCase):

    def test_valid_pos(self):
        # test function which checks if position is on the board
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        for m in range(0, size):
            for n in range(0, size):
                self.assertTrue(env.is_valid_pos((m, n)))

        self.assertFalse(env.is_valid_pos((-1, 0)))
        self.assertFalse(env.is_valid_pos((0, -1)))
        self.assertFalse(env.is_valid_pos((0, size)))
        self.assertFalse(env.is_valid_pos((size, 0)))
        self.assertFalse(env.is_valid_pos((size, size)))

    def test_can_move_to_pos(self):
        # test function which checks if position is on the board
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)

        # can move nowhere
        env.stones = np.full((size, size), True)
        for m in range(-1, size+1):
            for n in range(-1, size+1):
                self.assertFalse(env.can_move_to_pos((m, n)))

        # can move everywhere
        env.stones = np.full((size, size), False)
        for m in range(0, size):
            for n in range(0, size):
                self.assertTrue(env.can_move_to_pos((m, n)))

        self.assertFalse(env.can_move_to_pos((-1, 0)))
        self.assertFalse(env.can_move_to_pos((0, -1)))
        self.assertFalse(env.can_move_to_pos((0, size)))
        self.assertFalse(env.can_move_to_pos((size, 0)))
        self.assertFalse(env.can_move_to_pos((size, size)))

    def test_bomb_3x3(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)

        # bomb upper left corner
        env.stones = np.full((size, size), True)
        self.assertEqual(env.bomb_3x3((0,0)), 4)
        for m in range(0, size):
            for n in range(0, size):
                if m < 2 and n < 2:
                    self.assertFalse(env.stones[(m,n)])
                else:
                    self.assertTrue(env.stones[(m,n)])

        # bomb 1, 1
        env.stones = np.full((size, size), True)
        self.assertEqual(env.bomb_3x3((1, 1)), 9)
        for m in range(0, size):
            for n in range(0, size):
                if m < 3 and n < 3:
                    self.assertFalse(env.stones[(m, n)])
                else:
                    self.assertTrue(env.stones[(m, n)])

    def test_reset(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.reset()

        # check that no stones around agent but everywhere else
        board = env.make_observation_2D()
        a_m, a_n = env.agent_pos
        for m in range(0, size):
            for n in range(0, size):
                l2_dist_to_agent = math.sqrt((a_m - m)**2 + (a_n - n)**2)
                if l2_dist_to_agent < 1.0:
                    self.assertAlmostEqual(board[(m, n)], env.agent_val)
                    self.assertFalse(env.stones[(m, n)])
                elif l2_dist_to_agent < 2.0:
                    self.assertAlmostEqual(board[(m, n)], env.empty_val)
                    self.assertFalse(env.stones[(m, n)])
                else:
                    self.assertAlmostEqual(board[(m, n)], env.rock_val)
                    self.assertTrue(env.stones[(m, n)])

    def test_move_actions(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.set_initial_board((0,0))

        # agent at (0,0) -> can initially move only to 3 bording squares. Others are rocks or wall.
        obs, reward, _, _, _ = env.step(0) # up not possible
        self.assertAlmostEqual(reward, -1.0)
        self.assertAlmostEqual(env.make_observation_2D()[(0,0)], env.agent_val)

        obs, reward, _, _, _ = env.step(3)  # left not possible
        self.assertAlmostEqual(reward, -1.0)

        obs, reward, _, _, _ = env.step(1)  # right possible
        self.assertAlmostEqual(reward, -0.2)
        self.assertEqual(env.agent_pos, (0,1))
        self.assertAlmostEqual(env.make_observation_2D()[(0, 0)], env.empty_val) # previous field empty
        self.assertAlmostEqual(env.make_observation_2D()[(0, 1)], env.agent_val) # current field agent

        obs, reward, _, _, _ = env.step(1)  # right again not possible
        self.assertAlmostEqual(reward, -1.0)
        self.assertEqual(env.agent_pos, (0, 1))

        obs, reward, _, _, _ = env.step(2)  # down possible
        self.assertAlmostEqual(reward, -0.2)
        self.assertEqual(env.agent_pos, (1, 1))
        self.assertAlmostEqual(env.make_observation_2D()[(0, 1)], env.empty_val)  # previous field empty
        self.assertAlmostEqual(env.make_observation_2D()[(1, 1)], env.agent_val)  # current field agent

        obs, reward, _, _, _ = env.step(2)  # down again not possible
        self.assertAlmostEqual(reward, -1.0)
        self.assertEqual(env.agent_pos, (1, 1))

        obs, reward, _, _, _ = env.step(3)  # left possible
        self.assertAlmostEqual(reward, -0.2)
        self.assertEqual(env.agent_pos, (1, 0))

        obs, reward, _, _, _ = env.step(3)  # left again not possible
        self.assertAlmostEqual(reward, -1.0)
        self.assertEqual(env.agent_pos, (1, 0))

        obs, reward, _, _, _ = env.step(0)  # up possible
        self.assertAlmostEqual(reward, -0.2)
        self.assertEqual(env.agent_pos, (0, 0))

    def test_bomb_actions(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.set_initial_board((0, 0))

        obs, reward, _, _, _ = env.step(4)  # no rock bombed
        self.assertAlmostEqual(reward, -1.0)
        self.assertAlmostEqual(env.make_observation_2D()[(0, 0)], env.agent_val)

        obs, reward, _, _, _ = env.step(1) # move to (0,1)
        obs, reward, _, _, _ = env.step(4)  # 2 rocks bombed
        self.assertAlmostEqual(reward, 1.0)

        obs, reward, _, _, _ = env.step(2)  # move to (1,1)
        obs, reward, _, _, _ = env.step(4)  # 3 rocks bombed
        self.assertAlmostEqual(reward, 2.0)

    def test_reach_target(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.set_initial_board((0, 0))

        # destroy all rocks except one
        env.stones.fill(False)
        env.stones[(0, 1)] = True

        obs, reward, terminated, _, _ = env.step(2)  # down
        self.assertAlmostEqual(reward, env.move_penalty)
        self.assertFalse(terminated)

        obs, reward, terminated, _, _ = env.step(4)  # bomb and all is destroyed
        self.assertAlmostEqual(reward, env.end_game_reward)
        self.assertTrue(terminated)

    def test_reach_max(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.reset()

        for i in range(0, 101):
            _, _, terminated, maxreached, _ = env.step(4)
            self.assertFalse(maxreached)
            self.assertFalse(terminated)

        _, _, terminated, maxreached, _ = env.step(4)
        self.assertTrue(maxreached)
        self.assertFalse(terminated)

    def test_good_run(self):
        reward = 0.0
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.set_initial_board((1, 1))

        for i in range(0, 7):
            _, r, terminated, _, _ = env.step(2) # down
            reward += r
            _, r, terminated, _, _ = env.step(4)  # bomb
            reward += r

        for i in range(0, 7):
            _, r, terminated, _, _ = env.step(1) # right
            reward += r
            _, r, terminated, _, _ = env.step(4)  # bomb
            reward += r

        for i in range(0, 7):
            _, r, terminated, _, _ = env.step(0) # up
            reward += r
            _, r, terminated, _, _ = env.step(4)  # bomb
            reward += r

        for i in range(0, 4):
            _, r, terminated, _, _ = env.step(3)  # left
            reward += r
            _, r, terminated, _, _ = env.step(4)  # bomb
            reward += r

        for i in range(0, 4):
            _, r, terminated, _, _ = env.step(2) # down
            reward += r
            _, r, terminated, _, _ = env.step(4)  # bomb
            reward += r

        for i in range(0, 1):
            _, r, terminated, _, _ = env.step(1) # right
            self.assertFalse(terminated)
            reward += r
            _, r, terminated, _, _ = env.step(4)  # bomb
            self.assertFalse(terminated)
            reward += r

        _, r, terminated, _, _ = env.step(0)  # up
        reward += r
        self.assertFalse(terminated)
        _, r, terminated, _, _ = env.step(4)  # bomb
        reward += r
        self.assertTrue(terminated)
        print(reward)


    def test_destructable_agent(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100, indestructible_agent=False)
        env.set_initial_board((1, 1))

        _, r, _, _, _ = env.step(4)  # bomb at (1,1) and stay there at detonation
        self.assertAlmostEqual(r, env.bomb_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(0)  # up
        self.assertAlmostEqual(r, env.move_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(2)  # down and bomb detonates
        self.assertEqual(env.agent_pos, (1,1))
        self.assertAlmostEqual(r, env.move_penalty + env.close_bomb_penalty)
        self.assertEqual(len(env.active_bombs), 0)

        _, r, _, _, _ = env.step(4)  # bomb at (1,1) and stay (0,0) at detonation
        self.assertAlmostEqual(r, env.bomb_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(0)  # up
        self.assertAlmostEqual(r, env.move_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(3)  # left and bomb detonates
        self.assertEqual(env.agent_pos, (0, 0))
        self.assertAlmostEqual(r, env.move_penalty + env.close_bomb_penalty)
        self.assertEqual(len(env.active_bombs), 0)

        env.step(2)
        env.step(1)
        _, r, _, _, _ = env.step(4)  # bomb at (1,1) and stay (2,2) at detonation
        self.assertAlmostEqual(r, env.bomb_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(2)  # down
        self.assertAlmostEqual(r, env.move_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(1)  # right and bomb detonates
        self.assertEqual(env.agent_pos, (2, 2))
        self.assertAlmostEqual(r, env.move_penalty + env.close_bomb_penalty)
        self.assertEqual(len(env.active_bombs), 0)

        # drop bomb at (2,2) and go to safe place (0,2)
        _, r, _, _, _ = env.step(4)  # bomb at (1,1) and stay (2,2) at detonation
        self.assertAlmostEqual(r, env.bomb_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(0)  # up
        self.assertAlmostEqual(r, env.move_penalty)
        self.assertEqual(len(env.active_bombs), 1)
        _, r, _, _, _ = env.step(0)  # up
        self.assertEqual(env.agent_pos, (0, 2))
        self.assertAlmostEqual(r, env.move_penalty + 5 * env.rock_reward)
        self.assertEqual(len(env.active_bombs), 0)


    def test_if_bomb_on_field(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100, indestructible_agent=False)
        env.set_initial_board((1, 1))
        env.step(4)
        self.assertTrue(env.is_active_bomb_on_field((1,1)))
        self.assertFalse(env.is_active_bomb_on_field((0, 1)))
        self.assertFalse(env.is_active_bomb_on_field((0, 0)))
        self.assertFalse(env.is_active_bomb_on_field((1, 0)))
        self.assertFalse(env.is_active_bomb_on_field((2, 2)))
        self.assertFalse(env.is_active_bomb_on_field((2, 1)))


    def test_destructable_obs_bombs(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100, indestructible_agent=False)
        env.set_initial_board((1, 1))

        self.assertEqual(env.agent_pos, (1,1))
        env.step(4) # drop bomb, agent and bomb in same spot
        self.assertAlmostEqual(env.make_observation_2D()[(1,1)], env.bomb_and_agent_val )

        env.step(0)  # up, agent at (0,1) and bomb at (1,1)
        self.assertEqual(env.agent_pos, (0, 1))
        self.assertAlmostEqual(env.make_observation_2D()[(1, 1)], env.bomb_val)
        self.assertAlmostEqual(env.make_observation_2D()[(0, 1)], env.agent_val)

        env.step(2)  # down, agent at (1,1) and bomb detonates at (1,1)
        self.assertEqual(env.agent_pos, (1, 1))
        self.assertAlmostEqual(env.make_observation_2D()[(0, 1)], env.empty_val)
        self.assertAlmostEqual(env.make_observation_2D()[(1, 1)], env.agent_val)

    def test_destructable_multiple_bombs(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 150, indestructible_agent=False)
        env.set_initial_board((1, 1))

        self.assertEqual(env.agent_pos, (1,1))
        env.step(4) # drop bomb, agent and bomb in same spot
        self.assertAlmostEqual(env.make_observation_2D()[(1,1)], env.bomb_and_agent_val )
        self.assertEqual(len(env.active_bombs), 1)
        env.step(4)
        self.assertAlmostEqual(env.make_observation_2D()[(1, 1)], env.bomb_and_agent_val)
        self.assertEqual(len(env.active_bombs), 2)

        env.step(0) # up
        self.assertAlmostEqual(env.make_observation_2D()[(1, 1)], env.bomb_val)
        self.assertEqual(len(env.active_bombs), 1) # first bomb detonated

        env.step(1)  # right
        self.assertAlmostEqual(env.make_observation_2D()[(1, 1)], env.empty_val)
        self.assertEqual(len(env.active_bombs), 0)  # first bomb detonated

    def test_destructable_good_run(self):
        reward = 0.0
        size = 10
        env = bomberworld.BomberworldEnv(size, 100, indestructible_agent=False)
        env.set_initial_board((1, 1))

        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r

        self.assertAlmostEqual(3*env.move_penalty + 1 * env.bomb_penalty + 3.0*env.rock_reward, reward)

        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        self.assertAlmostEqual(5 * env.move_penalty + 2 * env.bomb_penalty + 3.0*env.rock_reward, reward)
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r

        self.assertAlmostEqual(7 * env.move_penalty + 2 * env.bomb_penalty + 6.0*env.rock_reward, reward)
        self.assertEqual(len(env.active_bombs), 0)

        _, r, _, _, _ = env.step(4)  # bomb + previous bomb explodes
        self.assertAlmostEqual(env.bomb_penalty, r)
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        self.assertAlmostEqual(env.move_penalty, r)
        reward += r
        _, r, _, _, _ = env.step(3)  # left + bomb exploding
        self.assertAlmostEqual(env.move_penalty+4*env.rock_reward, r)
        reward += r

        self.assertAlmostEqual(9 * env.move_penalty + 3 * env.bomb_penalty + 10.0*env.rock_reward, reward)

        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb + previous bomb explodes
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r

        self.assertAlmostEqual(13 * env.move_penalty + 4 * env.bomb_penalty + 14.0 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r

        self.assertAlmostEqual(16 * env.move_penalty + 5 * env.bomb_penalty + 18.0 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r

        self.assertAlmostEqual(23 * env.move_penalty + 7 * env.bomb_penalty + 26.0 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r

        self.assertAlmostEqual(29 * env.move_penalty + 8 * env.bomb_penalty + 31.0 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r

        self.assertAlmostEqual(35 * env.move_penalty + 10 * env.bomb_penalty + 39.0 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r


        self.assertAlmostEqual(43 * env.move_penalty + 12 * env.bomb_penalty + 48 * env.rock_reward, reward)


        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r

        self.assertAlmostEqual(48 * env.move_penalty + 14 * env.bomb_penalty + 55 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r

        self.assertAlmostEqual(53 * env.move_penalty + 16 * env.bomb_penalty + 62 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r

        self.assertAlmostEqual(59 * env.move_penalty + 17 * env.bomb_penalty + 67 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r

        self.assertAlmostEqual(63 * env.move_penalty + 18 * env.bomb_penalty + 71 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r

        self.assertAlmostEqual(65 * env.move_penalty + 19 * env.bomb_penalty + 75 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r

        self.assertAlmostEqual(75 * env.move_penalty + 21 * env.bomb_penalty + 79 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r
        _, r, _, _, _ = env.step(0)  # up
        reward += r

        self.assertAlmostEqual(81 * env.move_penalty + 22 * env.bomb_penalty + 84 * env.rock_reward, reward)

        _, r, _, _, _ = env.step(3)  # left
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(2)  # down
        reward += r
        _, r, _, _, _ = env.step(4)  # bomb
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r
        _, r, _, _, _ = env.step(1)  # right
        reward += r

        self.assertAlmostEqual(87 * env.move_penalty + 24 * env.bomb_penalty + 90 * env.rock_reward, reward)

        _, r, terminated, _, _ = env.step(4)  # bomb
        self.assertFalse(terminated)
        reward += r
        _, r, terminated, _, _ = env.step(1)  # right
        reward += r
        self.assertFalse(terminated)
        _, r, terminated, _, _ = env.step(1)  # right
        reward += r
        self.assertTrue(terminated)

        self.assertAlmostEqual(89 * env.move_penalty + 25 * env.bomb_penalty + 91 * env.rock_reward + env.end_game_reward, reward)

        print(reward)



if __name__ == '__main__':
    unittest.main()
