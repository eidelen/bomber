# do_bombing uses a checkpoint an runs a bomber game
# Author: Adrian Schneider

from ray.rllib.policy.policy import  Policy
import argparse
import bomberworld
from bomberworld_plotter import BomberworldPlotter

def run_bombing(path_to_checkpoint: str):

    trained_policy = Policy.from_checkpoint(path_to_checkpoint)

    env = bomberworld.BomberworldEnv(10, 20)
    o, info = env.reset()

    plotter = BomberworldPlotter(size=env.size) #, animated_gif_folder_path="gifs")
    plotter.add_frame(env.agent_pos, False, env.make_observation_2D())

    reward_sum = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        a = trained_policy.compute_single_action(o)[0]
        o, r, terminated, truncated, info = env.step(a)
        reward_sum += r
        plotter.add_frame(env.agent_pos, a == 4, env.make_observation_2D())
        plotter.plot_episode(current_reward=reward_sum)
        print("Current Reward:", reward_sum)

    print("Overall Reward:", reward_sum)
    plotter.create_animated_gif_from_episodes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='do_bombing',
        description='Runs bombin model')
    parser.add_argument('path', help='File path to checkpoint')
    args = parser.parse_args()
    run_bombing(args.path)
