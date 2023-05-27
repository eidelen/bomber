# Author: Adrian Schneider, armasuisse
# Note: Initial copied from Giacomo Del Rio, IDSIA

from datetime import datetime

import ray.tune
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env import EnvContext
from ray.tune import register_env, tune, Tuner
from ray.tune.stopper import MaximumIterationStopper

import bomberworld

def env_create(env_config: EnvContext):
    return bomberworld.BomberworldEnv(**env_config)

def print_ppo_configs(config):
    print("clip_param", config.clip_param)
    print("gamma", config.gamma)
    print("lr", config.lr)
    print("lamda", config.lambda_)

def grid_search_hypers(env_params: dict, nn_model: list, activation: str, desc: str, train_hw: dict):
    register_env("GridworldEnv", env_create)

    config = PPOConfig()

    print("Standard PPO Config:")
    print_ppo_configs(config)

    config = config.framework(framework='torch')
    config = config.resources(num_gpus=train_hw["gpu"])
    config = config.environment(env="GridworldEnv",
                                env_config=env_params)

    config.model['fcnet_hiddens'] = nn_model
    config.model['fcnet_activation'] = activation
    config.model['use_lstm'] = True,

    config = config.rollouts(num_rollout_workers=train_hw["cpu"])
    config = config.training( gamma=ray.tune.grid_search([0.75, 0.80, 0.85, 0.90, 0.95, 0.997])) # lr=ray.tune.grid_search([5e-05, 4e-05])) #, gamma=ray.tune.grid_search([0.99])) , lambda_=ray.tune.grid_search([1.0, 0.997, 0.95]))

    config = config.debugging(log_level="ERROR")

    experiment_name = f"PPO_{desc}_{datetime.now():%Y-%m-%d_%H-%M}_MODEL={nn_model}_ACT={activation}"

    tuner = Tuner(
        trainable=PPO,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir="out",
            verbose=2,
            stop=MaximumIterationStopper(200),
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=100)
        )
    )

    tuner.fit()


def resume_training():
    # restore checkpoint and resume learning
    # rsc: https://discuss.ray.io/t/unable-to-restore-fully-trained-checkpoint/8259/8
    # rsc: https://github.com/ray-project/ray/issues/4569
    # Note: The call works, but training does not continue (max iter reached?!)
    tuner = Tuner.restore("out/PPO_SmartBomber-DeadNearBomb-LongTraining-Gamma-0.75_2023-05-17_07-41_MODEL=[512, 512, 256, 128, 64]_ACT=relu")


    tuner.fit()


if __name__ == '__main__':

    #resume_training()

    if True:

        # train hw:
        hw = {"gpu": 0, "cpu": 3} # imac
        #hw = {"gpu": 1, "cpu": 11}  # adris

        env_params = {"size": 6, "max_steps": 40, "reduced_obs": True, "dead_when_colliding": True}
        #env_params = {"size": 10, "max_steps": 100, "indestructible_agent": False, "dead_near_bomb": True}
        # env_params = {"size": 10, "max_steps": 200, "dead_when_colliding": True, "dead_near_bomb": True, "indestructible_agent": False, "close_bomb_penalty": -1.0}
        nn_model = [256, 128, 64]
        activation = "relu"
        description = "ReducedBomber-Hyper"

        grid_search_hypers(env_params, nn_model, activation, description, hw)

