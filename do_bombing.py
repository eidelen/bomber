
from ray.rllib.policy.policy import  Policy
import bomberworld
from bomberworld_plotter import GridworldPlotter

def run_bombing():
    path_to_cp = "C:\\Users\\admin\\dev\\bomber\\out\\PPO_movepenalty_16-21_GAMMA=0.8_MODEL=[512, 256, 128, 64]_ACT=relu\\PPO_GridworldEnv_237d7_00000_0_2023-04-17_16-21-26\\checkpoint_000700\\policies\\default_policy"
    trained_policy = Policy.from_checkpoint(path_to_cp)

    env = bomberworld.GridworldEnv(10, 100)
    o, info = env.reset()

    plotter = GridworldPlotter(size=env.size)
    plotter.add_frame(env.agent_pos, False, env.board)

    reward_sum = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        a = trained_policy.compute_single_action(o)[0]
        o, r, terminated, truncated, info = env.step(a)
        plotter.add_frame(env.agent_pos, a == 4, env.board)
        reward_sum += r
        print(env.board)

        print("#################################################")

    print(reward_sum)
    plotter.plot_episode()


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
    run_bombing()