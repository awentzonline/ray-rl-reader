from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class RenderTextCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        if np.random.uniform() < 0.1:
            print(base_env.vector_env.envs[0].render())
