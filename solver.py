from datetime import datetime

from ray import air
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env import EnvContext
from ray.tune import register_env, tune, Tuner
from ray.tune.stopper import MaximumIterationStopper

import bomberworld

def env_create(env_config: EnvContext):
    return bomberworld.BomberworldEnv(**env_config)

def apply_ppo(gamma: float, nn_model: list, activation: str, desc: str):
    register_env("GridworldEnv", env_create)

    config = PPOConfig()
    config = config.framework(framework='torch')
    #config = config.resources(num_gpus=1)
    config = config.environment(env="GridworldEnv", env_config={"size": 10, "max_steps": 100, "indestructible_agent": False})

    config.model['fcnet_hiddens'] = nn_model
    config.model['fcnet_activation'] = activation

    #config = config.rollouts(num_rollout_workers=11)
    config = config.rollouts(num_rollout_workers=3)
    config = config.training(gamma=gamma) # not below 0.7
    config = config.debugging(log_level="ERROR")


    experiment_name = f"PPO_{desc}_{datetime.now():%H-%M}_GAMMA={gamma}_MODEL={nn_model}_ACT={activation}"

    tuner = Tuner(
        trainable=PPO,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir="out",
            verbose=2,
            stop=MaximumIterationStopper(2000),
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=100)
        )
    )

    tuner.fit()

def resume_training():
    # restore checkpoint and resume learning
    # rsc: https://discuss.ray.io/t/unable-to-restore-fully-trained-checkpoint/8259/8
    # rsc: https://github.com/ray-project/ray/issues/4569
    # Note: The call works, but training does not continue (max iter reached?!)
    tuner = Tuner.restore(
        "/Users/eidelen/dev/bomber/out/PPO_GRIDWORLD_15-52_GAMMA=0.8_MODEL=[256, 256, 128, 64]_ACT=relu")

    tuner
    tuner.fit()


if __name__ == '__main__':
    #resume_training()
    apply_ppo(0.8, [512, 256, 128, 64], "relu", "SuperBomber")

