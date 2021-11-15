from copy import deepcopy as copy

import gym
from gym import spaces
import numpy as np
import pandas as pd
from ray.rllib.env import MultiAgentEnv
import torch
from transformers import AutoTokenizer


class SentPieceMLM(MultiAgentEnv):
    """Learn to fill in masked tokens"""
    cursor_actions = [
        'move_left',
        'move_right',
        'set_token',
        'finished',
    ]

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, env_config):
        self.setup(env_config)
        self.corruption_rate = env_config.get('corruption_rate', 0.2)
        self.reward_exploration = env_config.get('reward_exploration', False)
        self.base_reward = env_config.get('base_reward', -0.1)
        self.informative_reward = env_config.get('informative_reward', True)

        self.mask_token_id = self.tokenizer.vocab[self.tokenizer.mask_token]
        self.vocab_size = len(self.tokenizer.vocab)
        self.cursor_action_space = spaces.Discrete(len(self.cursor_actions))
        self.token_action_space = spaces.Discrete(self.vocab_size)

        obs_dims = self.vocab_size
        print('observation dims', obs_dims)
        self.observation_space = spaces.Discrete(obs_dims)

        self.reset()

    def setup(self, conf):
        self.docs = self.load_docs(
            conf['corpus_path'], text_col=conf.get('text_col', 'text')
        )
        self.tokenizer = self.load_tokenizer(
            conf.get('hf_model_name', 'bert-base-uncased')
        )

    def reset(self):
        # corrupt the original tokens
        rand_doc_i = np.random.randint(len(self.docs))
        doc = self.docs[rand_doc_i]
        tokens = np.array(self.tokenizer.encode(doc, truncation=True))
        self.original_tokens = copy(tokens)
        # corrupt the tokens
        num_tokens = len(tokens)
        num_corrupted = max(int(num_tokens * self.corruption_rate), 1)
        corrupted_indicies = np.random.choice(
            num_tokens, num_corrupted, replace=False
        ).astype(np.int32)
        self.is_corrupt = np.zeros((num_tokens,)).astype(np.bool)
        self.is_corrupt[corrupted_indicies] = True

        tokens[corrupted_indicies] = self.mask_token_id
        self.tokens = tokens
        # reset env state
        self.marked_corrupt = np.zeros((len(tokens),)).astype(np.bool)
        self.visited = np.zeros((len(tokens),)).astype(np.bool)
        self.cursor = 0

        self.num_cursor_steps = 0
        self.token_agent_id = "token_{}".format(
            self.num_cursor_steps
        )

        return {
            'cursor_agent': self.current_obs()
        }

    def current_obs(self):
        return self.tokens[self.cursor]

    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "cursor_agent" in action_dict:
            self.num_cursor_steps += 1
            return self._cursor_step(action_dict['cursor_agent'])
        else:
            return self._token_step(list(action_dict.values())[0])

    def _cursor_step(self, action):
        name = self.cursor_actions[action]
        f_action = getattr(self, name)
        obs, reward, is_done = f_action()
        if not isinstance(reward, dict):
            reward += self.base_reward
            reward = {'cursor_agent': reward}
        is_done = {'__all__': is_done}
        return obs, reward, is_done, {}

    def set_token(self):
        """
        This is where the cursor agent tells the token agent
        to assign a token in the next step.
        """
        self.token_agent_id = "token_{}".format(
            self.num_cursor_steps
        )
        reward = {'token_agent': self.base_reward}
        return {self.token_agent_id: self.current_obs()}, reward, False

    def move_left(self):
        self.cursor = max(0, self.cursor - 1)
        return {'cursor_agent': self.current_obs()}, self.base_reward, False

    def move_right(self):
        self.cursor = min(len(self.tokens) - 1, self.cursor + 1)
        return {'cursor_agent': self.current_obs()}, self.base_reward, False

    def finished(self):
        num_corrupt = self.is_corrupt.sum()
        if num_corrupt == 0:
            num_corrupt = 1
        reward = -(self.tokens != self.original_tokens).sum() / num_corrupt
        # print(self.render())
        return {'cursor_agent': self.current_obs()}, reward + self.base_reward, True

    def _token_step(self, new_token_id):
        self.marked_corrupt[self.cursor] = True
        reward = self.base_reward
        if self.informative_reward:
            old_token_id = self.tokens[self.cursor]
            token_was_masked = self.is_corrupt[self.cursor]
            if old_token_id == new_token_id:
                reward = -1.
            else:
                reward += token_was_masked * 2 - 1
                new_token_is_correct = \
                    new_token_id == self.original_tokens[self.cursor]
                reward += new_token_is_correct * 2 - 1

        self.tokens[self.cursor] = new_token_id
        obs = {
            'cursor_agent': self.current_obs(),
            self.token_agent_id: self.current_obs(),
        }
        reward = {
            'cursor_agent': reward,
            self.token_agent_id: reward,
        }
        is_done = {'__all__': False, self.token_agent_id: True}

        return obs, reward, is_done, {}

    def render(self, *args, **kwargs):
        decoded = self.tokenizer.convert_ids_to_tokens(self.tokens)
        return ' '.join(decoded)

    def load_docs(self, path, text_col='text'):
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_pickle(path)
        text = df[text_col]
        return text

    def load_tokenizer(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
