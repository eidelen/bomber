from datetime import datetime

from ray import air
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env import EnvContext
from ray.tune import register_env, tune, Tuner
from ray.tune.stopper import MaximumIterationStopper

import bomberworld
from bomberworld_plotter import GridworldPlotter

def env_create(env_config: EnvContext):
    return bomberworld.GridworldEnv(**env_config)

def apply_ppo():
    register_env("GridworldEnv", env_create)

    config = PPOConfig()
    config = config.framework(framework='torch')
    config = config.environment(env="GridworldEnv", env_config={"size": 10, "max_steps": 100})
    config = config.rollouts(num_rollout_workers=5)
    config = config.training(gamma=0.9) # not below 0.7
    config = config.debugging(log_level="ERROR")

    experiment_name = f"PPO_GRIDWORLD_{datetime.now():%Y-%m-%d_%H-%M-%S}"

    tuner = Tuner(
        trainable=PPO,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            name=experiment_name,
            local_dir="out",
            verbose=2,
            stop=MaximumIterationStopper(1000),
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50)
        )
    )
    tuner.fit()

if __name__ == '__main__':
    apply_ppo()