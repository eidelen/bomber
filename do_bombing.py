# do_bombing uses a checkpoint an runs a bomber game
# Author: Adrian Schneider, armasuisse

from ray.rllib.policy.policy import  Policy
import numpy as np
import argparse
import bomberworld
from bomberworld_plotter import BomberworldPlotter

def run_bombing(path_to_checkpoint: str):

    trained_policy = Policy.from_checkpoint(path_to_checkpoint)
    model_config = trained_policy.model.model_config

    # hack to make lstm work
    cell_size = 256
    lstm_states = [np.zeros(cell_size, np.float32),
             np.zeros(cell_size, np.float32)]
    # end hack

    env = bomberworld.BomberworldEnv(6, 40, dead_when_colliding=True, reduced_obs=True)
    o, info = env.reset()

    plotter = BomberworldPlotter(size=env.size, animated_gif_folder_path="gifs")
    plotter.add_frame(env.agent_pos, None, None, env.make_current_board_2D())

    reward_sum = 0
    terminated, truncated = False, False
    while not (terminated or truncated):

        # Hack to make lstm work
        a, next_states, _ = trained_policy.compute_single_action(o, state=lstm_states)  # When using lstm -> "assert seq_lens is not None" : https://github.com/ray-project/ray/issues/10448#issuecomment-1151468435
        lstm_states = next_states

        o, r, terminated, truncated, info = env.step(a)
        reward_sum += r
        plotter.add_frame(agent_position=env.agent_pos, placed_bomb=info["placed_bomb"], exploded_bomb=info["exploded_bomb"], stones=env.make_current_board_2D())
        plotter.plot_episode(current_reward=reward_sum)
        print("Current Reward:", reward_sum)
    print("Overall Reward:", reward_sum)
    plotter.create_animated_gif_from_episodes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='do_bombing',
        description='Runs bombing model')
    parser.add_argument('path', help='File path to checkpoint')
    args = parser.parse_args()
    run_bombing(args.path)
