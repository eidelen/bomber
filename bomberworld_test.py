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
        env.board = np.zeros(shape=(size, size), dtype=np.float32)
        for m in range(-1, size+1):
            for n in range(-1, size+1):
                self.assertFalse(env.can_move_to_pos((m, n)))

        # can move everywhere
        env.board = np.ones(shape=(size, size), dtype=np.float32)
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
        env.board = np.zeros(shape=(size, size), dtype=np.float32)
        self.assertEqual(env.bomb_3x3((0,0)), 4)
        for m in range(0, size):
            for n in range(0, size):
                if m < 2 and n < 2:
                    self.assertAlmostEqual(env.board[(m,n)], env.empty_val)
                else:
                    self.assertAlmostEqual(env.board[(m,n)], env.rock_val)

        # bomb 1, 1
        env.board = np.zeros(shape=(size, size), dtype=np.float32)
        self.assertEqual(env.bomb_3x3((1, 1)), 9)
        for m in range(0, size):
            for n in range(0, size):
                if m < 3 and n < 3:
                    self.assertAlmostEqual(env.board[(m, n)], env.empty_val)
                else:
                    self.assertAlmostEqual(env.board[(m, n)], env.rock_val)

    def test_reset(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.reset()

        # check that no stones around agent but everywhere else
        a_m, a_n = env.agent_pos
        for m in range(0, size):
            for n in range(0, size):
                l2_dist_to_agent = math.sqrt((a_m - m)**2 + (a_n - n)**2)
                if l2_dist_to_agent < 1.0:
                    self.assertAlmostEqual(env.board[(m, n)], env.agent_val)
                elif l2_dist_to_agent < 2.0:
                    self.assertAlmostEqual(env.board[(m, n)], env.empty_val)
                else:
                    self.assertAlmostEqual(env.board[(m, n)], env.rock_val)


    def test_move_actions(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.set_initial_board((0,0))

        # agent at (0,0) -> can initially move only to 3 bording squares. Others are rocks or wall.
        obs, reward, _, _, _ = env.step(0) # up not possible
        self.assertAlmostEqual(reward, -1.0)

        obs, reward, _, _, _ = env.step(3)  # left not possible
        self.assertAlmostEqual(reward, -1.0)

        obs, reward, _, _, _ = env.step(1)  # right possible
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(env.agent_pos, (0,1))

        obs, reward, _, _, _ = env.step(1)  # right again not possible
        self.assertAlmostEqual(reward, -1.0)
        self.assertEqual(env.agent_pos, (0, 1))

        obs, reward, _, _, _ = env.step(2)  # down possible
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(env.agent_pos, (1, 1))

        obs, reward, _, _, _ = env.step(2)  # down again not possible
        self.assertAlmostEqual(reward, -1.0)
        self.assertEqual(env.agent_pos, (1, 1))

        obs, reward, _, _, _ = env.step(3)  # left possible
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(env.agent_pos, (1, 0))

        obs, reward, _, _, _ = env.step(3)  # left again not possible
        self.assertAlmostEqual(reward, -1.0)
        self.assertEqual(env.agent_pos, (1, 0))

        obs, reward, _, _, _ = env.step(0)  # up possible
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(env.agent_pos, (0, 0))

    def test_bomb_actions(self):
        size = 10
        env = bomberworld.BomberworldEnv(size, 100)
        env.set_initial_board((0, 0))

        obs, reward, _, _, _ = env.step(4)  # no rock bombed
        self.assertAlmostEqual(reward, -1.0)

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

        # destroy all rocks
        env.board.fill(128)
        env.board[(0, 0)] = 255
        env.board[(0, 1)] = 0

        obs, reward, terminated, _, _ = env.step(2)  # down
        self.assertAlmostEqual(reward, 0.0)
        self.assertFalse(terminated)

        obs, reward, terminated, _, _ = env.step(4)  # bomb and all is destroyed
        self.assertAlmostEqual(reward, 10.0)
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






        # plotter = GridworldPlotter(size=env.size, goal_pos=env.goal_pos)
        #
        # s = [1, 2, 1, 2]
        # for a in s:
        #     env.step(a)
        #     plotter.add_frame(env.agent_pos)
        #
        # plotter.plot_episode()
        #
        # env.agent_pos = (0, 0)
        #
        # # hit upper border
        # o, reward, terminated, truncate, info = env.step(0)
        # self.assertAlmostEqual(-1.1, reward, 0.00001)
        # self.assertEqual(env.agent_pos, (0, 0))


    # def test_use_trained_net(self):
    #     from ray.rllib.policy.policy import  Policy
    #
    #     trained_policy = Policy.from_checkpoint("/Users/eidelen/dev/CASLive/out/PPO_GRIDWORLD_2023-04-05_16-01-38/PPO_GridworldEnv_64017_00000_0_2023-04-05_16-01-41/checkpoint_000800/policies/default_policy")
    #
    #     env = gridworld.GridworldEnv(10, 100)
    #     o, info = env.reset()
    #
    #     plotter = GridworldPlotter(size=env.size, goal_pos=env.goal_pos)
    #
    #     reward_sum = 0
    #     terminated, truncated = False, False
    #     while not (terminated or truncated):
    #         a = trained_policy.compute_single_action(o)[0]
    #         o, r, terminated, truncated, info = env.step(a)
    #         plotter.add_frame(env.agent_pos)
    #         reward_sum += r
    #
    #     print(reward_sum)
    #     plotter.plot_episode()



if __name__ == '__main__':
    unittest.main()
