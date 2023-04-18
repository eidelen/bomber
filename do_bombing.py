
from ray.rllib.policy.policy import  Policy
import argparse
import bomberworld
from bomberworld_plotter import GridworldPlotter

def run_bombing(path_to_checkpoit: str):

    trained_policy = Policy.from_checkpoint(path_to_checkpoit)

    env = bomberworld.GridworldEnv(10, 100)
    o, info = env.reset()

    plotter = GridworldPlotter(size=env.size)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='do_bombing',
        description='Runs bombin model')
    parser.add_argument('path', help='File path to checkpoint')
    args = parser.parse_args()
    run_bombing(args.path)
