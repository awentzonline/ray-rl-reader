from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class SharedLSTM(nn.Module):
    def __init__(self, input_dims, latent_dims, rnn_layers=1, batch_first=False):
        super().__init__()
        self.rnn = nn.LSTM(
            input_dims, latent_dims, num_layers=rnn_layers,
            batch_first=batch_first
        )
        self.rnn_h0 = nn.Parameter(torch.randn(rnn_layers, 1, latent_dims))
        self.rnn_c0 = nn.Parameter(torch.randn(rnn_layers, 1, latent_dims))

    def initial_rnn_state(self, n=1):
        return (
            self.rnn_h0.repeat(1, n, 1),
            self.rnn_c0.repeat(1, n, 1),
        )

    def forward(self, x, hidden=None):
        if self.batch_first:
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[1]

        if hidden is None:
            hidden = self.initial_rnn_state(batch_size)

        y, hidden = self.rnn(x, hidden=hidden)
        return y, hidden


class SharedDecoder(TorchModelV2, nn.Module):
    def __init__(self, observation_space, action_space, num_outputs,
                 model_config, name):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs,
            model_config, name
        )
        nn.Module.__init__(self)

        latent_dims = model_config.get('latent_dims', 100)
        self.world_model = SharedLSTM(obs_dims, latent_dims)
        self.predict_policy = nn.Sequential(
            nn.Linear(latent_dims, num_outputs)
        )

    def forward(self, obs, hidden=None):
        world_state, hidden = self.world_model(obs, hidden=hidden)
        p_policy = self.predict_policy(world_state)
        return p_policy, hidden
