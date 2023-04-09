import unittest
import bomberworld
from bomberworld_plotter import GridworldPlotter

class MyTestCase(unittest.TestCase):

    def test_moving(self):
        env = bomberworld.GridworldEnv(10, 100)
        o, info = env.reset()

        env.agent_pos = (0, 0)
        env.goal_pos = (9, 9)

        plotter = GridworldPlotter(size=env.size, goal_pos=env.goal_pos)

        s = [1, 2, 1, 2]
        for a in s:
            env.step(a)
            plotter.add_frame(env.agent_pos)

        plotter.plot_episode()

        env.agent_pos = (0, 0)

        # hit upper border
        o, reward, terminated, truncate, info = env.step(0)
        self.assertAlmostEqual(-1.1, reward, 0.00001)
        self.assertEqual(env.agent_pos, (0, 0))


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
