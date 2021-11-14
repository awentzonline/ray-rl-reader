from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class WorldModelLSTM(nn.Module):
    def __init__(self, obs_dims, latent_dims):
        super().__init__()
        self.rnn = nn.LSTM(obs_dims, latent_dims)

    def forward(self, obs, hidden=None):
        y, hidden = self.rnn(obs, hidden=hidden)
        return y, hidden


class MLMPolicy(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs,
                 model_config, name):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs,
            model_config, name
        )
        nn.Module.__init__(self)

        latent_dims = model_config.get('latent_dims', 100)
        token_dims = observation_space.shape[-1]
        num_cursor_actions = num_outputs

        self.world_model = WorldModelLSTM(obs_dims, latent_dims)
        self.predict_cursor_policy = nn.Sequential(
            nn.Linear(latent_dims, num_cursor_actions)
        )
        self.predict_token = nn.Sequential(
            nn.Linear(latent_dims, token_dims)
        )

    def forward(self, obs, hidden=None):
        world_state, hidden = self.world_model(obs, hidden=hidden)
        p_cursor_policy = self.predict_cursor_policy(world_state)
        p_token = self.predict_token(world_state)
        return p_cursor_policy, p_token, hidden
