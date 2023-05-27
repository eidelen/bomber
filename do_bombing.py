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

    # hack to make lstm work -> does not work: 'PPOTorchPolicy' object is not subscriptable
    transformer_attention_size = model_config["attention_dim"]
    transformer_memory_size = model_config["attention_memory_inference"]
    transformer_layer_size = np.zeros([transformer_memory_size, transformer_attention_size])
    transformer_length = model_config["attention_num_transformer_units"]
    state_list = transformer_length * [transformer_layer_size]
    initial_state_list = state_list
    # end hack

    env = bomberworld.BomberworldEnv(6, 40, dead_when_colliding=True, reduced_obs=True)
    o, info = env.reset()

    plotter = BomberworldPlotter(size=env.size, animated_gif_folder_path="gifs")
    plotter.add_frame(env.agent_pos, None, None, env.make_observation_2D())

    reward_sum = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        a = trained_policy.compute_single_action(o)[0]  # When using lstm -> "assert seq_lens is not None" : https://github.com/ray-project/ray/issues/10448#issuecomment-1151468435
        o, r, terminated, truncated, info = env.step(a)
        reward_sum += r
        plotter.add_frame(agent_position=env.agent_pos, placed_bomb=info["placed_bomb"], exploded_bomb=info["exploded_bomb"], stones=env.make_observation_2D())
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
