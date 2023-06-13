# Author: Adrian Schneider, armasuisse
# Note: Initial copied from Giacomo Del Rio, IDSIA

from datetime import datetime

import torch.nn as nn
import torch

import ray.tune
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env import EnvContext
from ray.tune import register_env, tune, Tuner
from ray.tune.stopper import MaximumIterationStopper
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension

import bomberworld

class MyConvModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 16, (3, 3), 2)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), 1)
        self.flat1 = nn.Flatten()

        self.lin1 = nn.Linear(128, num_outputs)

        self.valuef = nn.Linear(128, 1)
        self.value = None

    def forward(self, input_dict, state, seq_lens):
        x = self.conv1(input_dict["obs"])
        x = self.conv2(x)
        x = self.flat1(x)
        act = self.lin1(x)

        self.value = self.valuef(x)

        return act, []

    def value_function(self):
        return torch.reshape(self.value, [-1])


class MyRNNModel(RecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 16, (3, 3), 2)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), 1)
        self.flat1 = nn.Flatten()

        self.rnn = nn.GRU(128, 128, batch_first=True)

        self.lin1 = nn.Linear(128, num_outputs)

        self.valuef = nn.Linear(128, 1)
        self.value = None

    def get_initial_state(self):
        return [torch.zeros(128)]

    def forward(self, input_dict, state, seq_lens):
        act, new_state = self.forward_rnn(input_dict, state, seq_lens)
        return torch.reshape(act, [-1, self.num_outputs]), new_state

    def forward_rnn(self, inputs, state, seq_lens):
        x = self.conv1(inputs["obs"])
        x = self.conv2(x)
        x = self.flat1(x)
        x, h = self.rnn(add_time_dimension(x, seq_lens=seq_lens, framework="torch", time_major=False),
                        torch.unsqueeze(state[0], 0))
        act = self.lin1(x.reshape(-1, 128))

        self.value = self.valuef(x.reshape(-1, 128))
        return act, [torch.squeeze(h, 0)]

    def value_function(self):
        return torch.reshape(self.value, [-1])

def env_create(env_config: EnvContext):
    return bomberworld.BomberworldEnv(**env_config)

def print_ppo_configs(config):
    print("Ray Version:", ray.__version__)
    print("clip_param", config.clip_param)
    print("gamma", config.gamma)
    print("lr", config.lr)
    print("lamda", config.lambda_)
    print("conv", config.model['conv_filters'])

def grid_search_hypers(env_params: dict, nn_fc_model: list, nn_cv_model: list, activation: str, desc: str, train_hw: dict, use_lstm: bool):
    register_env("GridworldEnv", env_create)

    config = PPOConfig()

    print("Standard PPO Config:")
    print_ppo_configs(config)

    config = config.framework(framework='torch')
    config = config.resources(num_gpus=train_hw["gpu"])
    config = config.environment(env="GridworldEnv",
                                env_config=env_params)

    ModelCatalog.register_custom_model("my_rnn_model", MyRNNModel)

    config.model['custom_model'] = "my_rnn_model"

    #config.model = {"dim": 10, "conv_filters":  [[16, [3, 3], 1], [32, [3, 3], 1], [64, [6, 6], 6]]}

    #
    # if nn_fc_model is not None:
    #     config.model['fcnet_hiddens'] = nn_fc_model
    #     config.model['fcnet_activation'] = activation
    #
    # if nn_cv_model is not None:
    #     config.model['dim'] = 3 # 3 in reduced mode
    #     config.model['conv_filters'] = nn_cv_model
    #
    #
    # if use_lstm:
    #     # another help -> https://github.com/ray-project/ray/issues/9220
    #     config.model['use_lstm'] = True
    #     # Max seq len for training the LSTM, defaults to 20.
    #     config.model['max_seq_len'] = 20
    #     # Size of the LSTM cell.
    #     config.model['lstm_cell_size'] = 256
    #     # Whether to feed a_{t-1}, r_{t-1} to LSTM.
    #     config.model['lstm_use_prev_reward'] = False
    #     config.model['lstm_use_prev_action'] = False

    config = config.rollouts(num_rollout_workers=train_hw["cpu"])
    config = config.training(gamma=ray.tune.grid_search([0.75, 0.8, 0.85, 0.90, 0.96, 0.99])) # lr=ray.tune.grid_search([5e-05, 4e-05])) #, gamma=ray.tune.grid_search([0.99])) , lambda_=ray.tune.grid_search([1.0, 0.997, 0.95]))

    config = config.debugging(log_level="ERROR")

    experiment_name = f"PPO_{desc}_{datetime.now():%Y-%m-%d_%H-%M}_MODEL={nn_fc_model}-{nn_cv_model}_ACT={activation}"

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
        #hw = {"gpu": 0, "cpu": 3} # imac
        hw = {"gpu": 1, "cpu": 11}  # adris

        env_params = {"size": 10, "max_steps": 130, "reduced_obs": False, "flatten_obs": False, "dead_when_colliding": True, "indestructible_agent": False, "dead_near_bomb": True}
        #env_params = {"size": 6, "max_steps": 60, "reduced_obs": True, "dead_when_colliding": True, "indestructible_agent": False, "dead_near_bomb": True}
        #env_params = {"size": 10, "max_steps": 100, "indestructible_agent": False, "dead_near_bomb": True}
        # env_params = {"size": 10, "max_steps": 200, "dead_when_colliding": True, "dead_near_bomb": True, "indestructible_agent": False, "close_bomb_penalty": -1.0}
        nn_fc_model = None #[256, 128, 64]
        nn_cv_model = None #[3, 3, 1]
        activation = "relu"
        description = "Hyper-ReducedConvo-6to8x6to8-LSTM"

        grid_search_hypers(env_params, nn_fc_model, nn_cv_model, activation, description, hw, use_lstm=True)

