import unittest
import numpy as np
import bomberworld
from bomberworld_plotter import GridworldPlotter

class MyTestCase(unittest.TestCase):

    def test_valid_pos(self):
        # test function which checks if position is on the board
        size = 10
        env = bomberworld.GridworldEnv(size, 100)
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
        env = bomberworld.GridworldEnv(size, 100)

        # can move nowhere
        env.board = np.zeros(shape=(size, size), dtype=np.uint8)
        for m in range(-1, size+1):
            for n in range(-1, size+1):
                self.assertFalse(env.can_move_to_pos((m, n)))

        # can move everywhere
        env.board = np.ones(shape=(size, size), dtype=np.uint8)
        for m in range(0, size):
            for n in range(0, size):
                self.assertTrue(env.can_move_to_pos((m, n)))

        self.assertFalse(env.can_move_to_pos((-1, 0)))
        self.assertFalse(env.can_move_to_pos((0, -1)))
        self.assertFalse(env.can_move_to_pos((0, size)))
        self.assertFalse(env.can_move_to_pos((size, 0)))
        self.assertFalse(env.can_move_to_pos((size, size)))

    def test_bomb_3x3(self):
        # test function which checks if position is on the board
        size = 10
        env = bomberworld.GridworldEnv(size, 100)

        # bomb upper left corner
        env.board = np.zeros(shape=(size, size), dtype=np.uint8)
        self.assertEqual(env.bomb_3x3((0,0)), 4)
        print(env.board)
        for m in range(0, size):
            for n in range(0, size):
                if m < 2 and n < 2:
                    self.assertTrue(env.board[(m,n)] == 128)
                else:
                    self.assertTrue(env.board[(m,n)] == 0)

        # bomb 1, 1
        env.board = np.zeros(shape=(size, size), dtype=np.uint8)
        self.assertEqual(env.bomb_3x3((1, 1)), 9)
        print(env.board)
        for m in range(0, size):
            for n in range(0, size):
                if m < 3 and n < 3:
                    self.assertTrue(env.board[(m, n)] == 128)
                else:
                    self.assertTrue(env.board[(m, n)] == 0)

    def test_reset(self):
        env = bomberworld.GridworldEnv(10, 100)
        o, info = env.reset()



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
